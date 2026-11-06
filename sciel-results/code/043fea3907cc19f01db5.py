"""Rewritten pebbling solver for communication-minimizing DAG pebbling."""
import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    """
    Solve the DAG pebbling problem to minimize communication (transfers).
    
    Args:
        dag: A dict with:
            - 'parents': dict mapping vertex -> list of parent vertices (topologically ordered)
            - 'inputs': list of input vertices (start blue)
            - 'outputs': list of output vertices (must finish blue)
        fast_memory_budget: int, number of fast memory slots available
        
    Returns:
        list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    # Extract DAG structure
    parents = dag['parents']
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    # Build children map
    children = {v: [] for v in parents}
    for v, ps in parents.items():
        for p in ps:
            if p not in children:
                children[p] = []
            children[p].append(v)
    
    # Topological order (parents are already topologically ordered per problem statement)
    # But let's ensure we have all vertices
    all_vertices = set(parents.keys())
    for v, ps in parents.items():
        for p in ps:
            all_vertices.add(p)
    
    # Create topological ordering
    in_degree = {v: 0 for v in all_vertices}
    for v, ps in parents.items():
        in_degree[v] = len(ps)
    
    topo_order = []
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # If topo_order doesn't include all vertices, add remaining (shouldn't happen in valid DAG)
    if len(topo_order) != len(all_vertices):
        for v in all_vertices:
            if v not in topo_order:
                topo_order.append(v)
    
    # Compute liveness: number of remaining children that need this vertex
    # A vertex is "live" if it has children that haven't been computed yet
    # We use a greedy strategy: keep vertices with high future dependency cost
    
    # Strategy: 
    # 1. Load all inputs
    # 2. Process vertices in topological order
    # 3. For each vertex, compute it if all parents are available (in memory or loaded)
    # 4. Manage memory by deleting vertices that are no longer needed (no remaining children)
    # 5. Store outputs at the end
    
    # Determine which vertices are "needed" - those that are ancestors of outputs
    needed = set(outputs)
    for v in reversed(topo_order):
        if v in needed:
            for p in parents.get(v, []):
                needed.add(p)
    
    # We only need to process 'needed' vertices
    # But we still need to handle memory for all computations
    
    # Greedy pebbling strategy:
    # - Maintain a set of vertices currently in fast memory (red pebbles)
    # - Process in topological order
    # - When computing a vertex, ensure all parents are in memory
    # - After computing, delete parents that are no longer needed (no other children pending)
    # - If memory is full, delete the vertex with lowest "liveness" (fewest remaining children)
    
    # Count remaining children for each vertex (children that still need to be computed)
    remaining_children = {v: len([c for c in children.get(v, []) if c in needed]) for v in needed}
    
    # Track which vertices are in memory
    in_memory = set()
    operations = []
    
    # Phase 1: Load inputs
    for v in inputs:
        if v in needed:
            operations.append(['load', v])
            in_memory.add(v)
    
    # Process vertices in topological order
    computed = set()
    for v in topo_order:
        if v not in needed:
            continue
            
        # If v is an input, it's already loaded
        if v in inputs:
            computed.add(v)
            # Update remaining children count for parents
            continue
        
        # Check if all parents are in memory
        ps = parents.get(v, [])
        parents_in_memory = all(p in in_memory for p in ps)
        
        if not parents_in_memory:
            # Need to load missing parents
            for p in ps:
                if p not in in_memory:
                    # Check if we can make room
                    if len(in_memory) >= fast_memory_budget:
                        # Delete a vertex with lowest liveness
                        # Prefer deleting vertices with no remaining children
                        candidates = [u for u in in_memory if u != v and u not in outputs]
                        if candidates:
                            # Choose vertex with minimum remaining children
                            min_lc = min(remaining_children.get(u, 0) for u in candidates)
                            to_delete = [u for u in candidates if remaining_children.get(u, 0) == min_lc]
                            # Among those, prefer one that's not an output
                            to_delete = [u for u in to_delete if u not in outputs] or to_delete
                            if to_delete:
                                del_v = to_delete[0]
                                operations.append(['delete', del_v])
                                in_memory.remove(del_v)
                        else:
                            # No candidates, we have a problem - but this shouldn't happen
                            # if budget is sufficient
                            pass
                    
                    # Now load the parent
                    operations.append(['load', p])
                    in_memory.add(p)
        
        # Compute v
        operations.append(['compute', v])
        in_memory.add(v)
        computed.add(v)
        
        # Update remaining children for parents
        for p in ps:
            if p in remaining_children:
                remaining_children[p] -= 1
        
        # Delete parents that are no longer needed
        for p in ps:
            if remaining_children.get(p, 0) == 0 and p not in outputs and p != v:
                if p in in_memory:
                    operations.append(['delete', p])
                    in_memory.remove(p)
        
        # If memory is over budget, delete vertices with lowest liveness
        while len(in_memory) > fast_memory_budget:
            # Find vertex to delete - not an output, not v
            candidates = [u for u in in_memory if u not in outputs and u != v]
            if not candidates:
                # Can't delete, break (shouldn't happen in valid cases)
                break
            min_lc = min(remaining_children.get(u, 0) for u in candidates)
            to_delete = [u for u in candidates if remaining_children.get(u, 0) == min_lc]
            # Prefer non-output vertices
            to_delete = [u for u in to_delete if u not in outputs] or to_delete
            if to_delete:
                del_v = to_delete[0]
                operations.append(['delete', del_v])
                in_memory.remove(del_v)
            else:
                break
    
    # Phase 2: Store outputs
    for v in outputs:
        if v in in_memory:
            operations.append(['store', v])
            in_memory.remove(v)
        else:
            # Need to reload and compute if not in memory
            # This is a fallback - ideally outputs should be in memory
            # For now, just load and store
            operations.append(['load', v])
            in_memory.add(v)
            operations.append(['store', v])
            in_memory.remove(v)
    
    # Cleanup remaining memory
    for v in list(in_memory):
        operations.append(['delete', v])
        in_memory.remove(v)
    
    return operations

# EVOLVE-BLOCK-END
