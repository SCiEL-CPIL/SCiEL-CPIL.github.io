# EVOLVE-BLOCK-START
import time
from collections import deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Args:
        dag: dict with 'vertices' (list), 'parents' (dict vertex->list of parents), 
             'outputs' (list of output vertices), 'inputs' (list of input vertices)
        fast_memory_budget: int, max red values in fast memory at once
    
    Returns:
        list of [operation, vertex] where operation is 'load', 'store', 'compute', 'delete'
    """
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    max_ops = 20000
    
    # Parse DAG structure
    vertices = list(dag.get('vertices', range(dag.get('n', 0))))
    n = len(vertices)
    parents = dict(dag.get('parents', {}))
    outputs = list(dag.get('outputs', []))
    inputs = set(dag.get('inputs', []))
    
    if n == 0:
        return []
    
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Build topological order
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    topo_order = []
    queue = deque([v for v in vertices if in_degree[v] == 0])
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # Map vertex to topo position
    topo_pos = {v: i for i, v in enumerate(topo_order)}
    
    # Count fan-out (number of children)
    fanout = {v: len(children[v]) for v in vertices}
    
    # Identify which vertices are inputs (no parents)
    input_vertices = [v for v in topo_order if len(parents.get(v, [])) == 0]
    
    # Strategy: Use a greedy approach with careful memory management
    # Key: minimize total transfers (load + store + delete)
    
    ops = []
    fast_mem = set()  # vertices currently in fast memory (red)
    
    # Track when a vertex's value is last needed
    # A vertex v is needed until all its children have been computed
    # We can delete v after all children that need v have been computed
    
    # For each vertex, determine the "last needed" time
    # This is the maximum topo position among its children
    last_needed = {}
    for v in vertices:
        if children[v]:
            last_needed[v] = max(topo_pos[c] for c in children[v])
        else:
            last_needed[v] = topo_pos[v]  # needed at least until computed
    
    # Process in topological order
    # For each vertex:
    # 1. If it's an input, load it if not in memory
    # 2. If it's not an input, compute it if all parents are in memory
    # 3. After computing, check if we can delete any vertices
    
    # First pass: determine which vertices we can keep in memory
    # We want to keep vertices with high fanout or that are needed later
    
    # Greedy: process vertices in topo order, maintaining memory budget
    for idx, v in enumerate(topo_order):
        if time.time() > deadline or len(ops) >= max_ops:
            break
        
        # If this is an input vertex, load it
        if v in inputs:
            if v not in fast_mem:
                ops.append(['load', v])
                fast_mem.add(v)
            continue
        
        # Non-input vertex: need to compute it
        # Check if all parents are in fast memory
        parent_list = parents.get(v, [])
        if not all(p in fast_mem for p in parent_list):
            # Some parent not in memory - need to reload or recompute
            # For simplicity, reload missing parents
            for p in parent_list:
                if p not in fast_mem:
                    if p in inputs:
                        ops.append(['load', p])
                    else:
                        # Recompute parent
                        pp = parents.get(p, [])
                        if all(q in fast_mem for q in pp):
                            ops.append(['compute', p])
                        else:
                            # Recursive case - handle by loading all ancestors
                            # This is a simplification; in practice, we'd need backtracking
                            for q in pp:
                                if q not in fast_mem:
                                    if q in inputs:
                                        ops.append(['load', q])
                                    else:
                                        ops.append(['compute', q])
                        fast_mem.add(p)
        
        # Compute v
        ops.append(['compute', v])
        fast_mem.add(v)
        
        # Check memory budget - if over, delete something
        while len(fast_mem) > fast_memory_budget:
            # Find vertex to delete
            # Priority: delete vertices that are no longer needed (all children processed)
            # or have lowest "importance" (low fanout, early last_needed)
            
            candidates = []
            for u in fast_mem:
                if u in inputs:
                    # Don't delete inputs unless absolutely necessary
                    continue
                # Check if u is still needed
                still_needed = False
                for c in children[u]:
                    if topo_pos[c] >= idx:  # child not yet processed
                        still_needed = True
                        break
                if not still_needed:
                    candidates.append((0, u))  # priority 0: not needed
                else:
                    # Still needed - lower priority for deletion
                    candidates.append((1, u))
            
            if not candidates:
                # All are inputs or still needed - delete an input as last resort
                for u in fast_mem:
                    if u in inputs:
                        candidates.append((2, u))
                        break
            
            if candidates:
                candidates.sort()
                del_v = candidates[0][1]
                ops.append(['delete', del_v])
                fast_mem.remove(del_v)
            else:
                break
    
    # Ensure all outputs are in fast memory
    for v in outputs:
        if time.time() > deadline or len(ops) >= max_ops:
            break
        if v not in fast_mem:
            if v in inputs:
                ops.append(['load', v])
                fast_mem.add(v)
            else:
                parent_list = parents.get(v, [])
                if all(p in fast_mem for p in parent_list):
                    ops.append(['compute', v])
                    fast_mem.add(v)
    
    # Store all outputs
    for v in outputs:
        if time.time() > deadline or len(ops) >= max_ops:
            break
        ops.append(['store', v])
    
    return ops
# EVOLVE-BLOCK-END
