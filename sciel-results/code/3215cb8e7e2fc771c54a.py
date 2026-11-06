"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque, defaultdict

def _topological_sort(vertices, parents):
    """
    Returns a list of vertices in topological order.
    vertices: list of vertex IDs
    parents: dict mapping vertex -> list of parent IDs
    """
    in_degree = {v: 0 for v in vertices}
    children = {v: [] for v in vertices}
    
    for v in vertices:
        if v in parents:
            for p in parents[v]:
                if p in in_degree:
                    in_degree[v] += 1
                    children[p].append(v)
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(order) != len(vertices):
        return vertices
    return order

def _compute_liveness(dag, order):
    """
    Precompute for each vertex v:
    - children[v]: list of children
    - last_use[v]: the index in 'order' of the last child of v that needs v.
      If v has no children, last_use[v] is the index of v itself (or -1 if not needed after compute).
      Actually, we need to know when v is no longer needed.
      A vertex v is needed until all its children have been computed.
      Let last_child_idx[v] = max(index of child c in order) for c in children[v].
      If v has no children, it's only needed until it's computed (or stored if output).
    """
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
            
    # Map vertex to its index in the topological order
    v_index = {v: i for i, v in enumerate(order)}
    
    last_use = {}
    for v in vertices:
        if children[v]:
            last_use[v] = max(v_index[c] for c in children[v])
        else:
            # Leaf node. If it's an output, we need to store it.
            # If it's an input, it's loaded and then... wait, inputs are leaves in reverse?
            # No, inputs have no parents. Outputs are specific vertices.
            # An input vertex v: needs to be in memory until all its children are computed.
            # An output vertex v: needs to be in memory until it is stored.
            # For a leaf that is not an output, it can be deleted immediately after compute?
            # Or is it an input? Inputs are sources.
            # Let's assume "last_use" for a vertex v is the index of the last operation that requires v.
            # If v is computed at index i, it is available.
            # It is required for child c at index j > i.
            # So last_use[v] = max(j for c in children[v]) if children exist.
            # If no children, last_use[v] = i (index of v itself) if we consider it "used" to be computed/stored.
            # Actually, if v has no children, it is only needed to be computed (if not input) or loaded (if input).
            # If v is an output, it must be stored.
            # Let's define last_use[v] as the latest index in the schedule where v must be in memory.
            # For v at index i:
            #   If v has children, last_use[v] = max(index(c) for c in children[v]).
            #   If v has no children, last_use[v] = i.
            last_use[v] = i if not children[v] else max(v_index[c] for c in children[v])
            
    return children, last_use

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    order = _topological_sort(vertices, parents)
    children, last_use = _compute_liveness(dag, order)
    
    # Identify inputs (no parents)
    inputs = [v for v in vertices if not parents.get(v, [])]
    
    # Strategy:
    # 1. Load all inputs.
    # 2. Process vertices in topological order.
    # 3. For each vertex v:
    #    - If v is an input, it's already in memory.
    #    - Else, ensure all parents are in memory. If a parent was deleted, we have a problem.
    #      To avoid this, we should not delete a vertex until it is no longer needed.
    #      "No longer needed" means last_use[parent] < current_index.
    #    - Compute v.
    #    - If v is an output, store it immediately (to free memory? or keep it?).
    #      "Outputs must finish blue". Storing moves it to slow memory (blue).
    #      Once stored, it's no longer in fast memory (red).
    #      So storing an output frees a fast-memory slot.
    #    - Check if any vertices can be deleted.
    #      A vertex u can be deleted if last_use[u] < current_index AND u is not an output (already stored) 
    #      AND u is not needed for future computations.
    #      Wait, if u is an output, we store it. After store, it's blue. We don't need it in red anymore.
    #      So outputs are "deleted" from red memory by the store operation.
    #      Non-output vertices are deleted by the delete operation.
    
    ops = []
    in_memory = set()
    
    # 1. Load inputs
    for v in inputs:
        ops.append(['load', v])
        in_memory.add(v)
        
    # Check if we even fit inputs in memory
    if len(in_memory) > fast_memory_budget:
        # This is a hard fail for simple strategy. 
        # But usually inputs are small or budget is sufficient for inputs.
        # If not, we might need to load, use, delete inputs immediately?
        # But inputs are needed for multiple children potentially.
        # For now, assume inputs fit. If not, this simple strategy fails.
        # We could implement a more complex load-on-demand for inputs, but that requires 
        # knowing if an input is needed again.
        # Let's stick to the simple strategy first.
        pass

    # 2. Process rest
    for i, v in enumerate(order):
        if v in inputs:
            continue
            
        # Ensure parents are in memory
        # In a strict topological pass with "delete only when no longer needed",
        # parents should always be in memory when we compute their child.
        # Let's verify:
        # Parent p of v. last_use[p] >= index(v) because v is a child of p.
        # So p should not have been deleted before index(v).
        
        # Compute v
        ops.append(['compute', v])
        in_memory.add(v)
        
        # If v is an output, store it
        if v in outputs:
            ops.append(['store', v])
            in_memory.remove(v)
            
        # Now, delete any vertices that are no longer needed
        # A vertex u in in_memory can be deleted if last_use[u] < i 
        # (meaning all its children have been processed) 
        # AND u is not an output (outputs are already stored and removed from in_memory).
        # Wait, if u is an output, it was removed from in_memory upon store.
        # So we only delete non-output intermediates.
        
        # We can delete multiple vertices at once to free space.
        # But we should only delete if we need to? Or always delete to keep memory low?
        # Deleting is an operation. Does it count towards "transfers"?
        # "Score mean transfers / naive schedule".
        # Usually transfers = loads + stores. Deletes are often free or low cost.
        # However, the task says "at most 20000 operations".
        # If deletes count as operations, we want to minimize them too?
        # But we MUST delete to free space for new computes if budget is tight.
        # If budget is loose, we can skip deletes? No, "every red value consumes one fast-memory slot".
        # If we don't delete, we might run out of slots.
        # So we MUST delete when necessary.
        
        # Optimization: Only delete if we are close to budget limit?
        # Or delete immediately to keep memory minimal?
        # Deleting immediately minimizes memory usage, which is safe.
        # It might increase operation count if deletes count against the 20000 limit.
        # But 20000 is likely large enough for these DAGs.
        
        # Let's delete all vertices that are no longer needed.
        to_delete = []
        for u in list(in_memory):
            if last_use[u] < i:
                # u is no longer needed for any future compute
                # u is not an output (outputs are not in in_memory)
                to_delete.append(u)
                
        for u in to_delete:
            ops.append(['delete', u])
            in_memory.remove(u)
            
        # Check memory limit
        if len(in_memory) > fast_memory_budget:
            # We are over budget. This should not happen if we delete aggressively.
            # If it does, we have a bug in liveness calculation or budget is too small for active set.
            # In that case, we might need to recompute, but that's complex.
            # For now, we assume aggressive deletion keeps us within budget.
            pass

    # 3. Cleanup
    # Any remaining vertices in in_memory must be handled.
    # If they are outputs, they should have been stored.
    # If they are not outputs, they should have been deleted.
    # But just in case, delete any remaining non-outputs.
    for u in list(in_memory):
        if u not in outputs:
            ops.append(['delete', u])
            in_memory.remove(u)
        else:
            # Output still in memory? Should have been stored.
            # If it wasn't stored (maybe I missed it?), store it now.
            ops.append(['store', u])
            in_memory.remove(u)

    # Final check: all outputs must be blue.
    # We stored them when computed.
    
    return ops

# EVOLVE-BLOCK-END
