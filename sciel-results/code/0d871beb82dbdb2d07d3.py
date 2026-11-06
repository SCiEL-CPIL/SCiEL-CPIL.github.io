"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    """
    Minimize memory transfers in DAG pebbling using lookahead-based eviction.
    """
    from collections import deque
    
    # Extract DAG structure
    vertices = dag.get('vertices', list(dag.get('parents', {}).keys()))
    parents = dag.get('parents', {})
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    # Build children map and compute topological order
    children = {v: [] for v in vertices}
    in_degree = {v: 0 for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
            in_degree[v] += 1
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        topo_order = vertices
    
    vertex_pos = {v: i for i, v in enumerate(topo_order)}
    
    # For each vertex, precompute the positions in topo_order where it's needed as a parent
    needed_at = {v: [] for v in vertices}
    for i, v in enumerate(topo_order):
        for p in parents.get(v, []):
            needed_at[p].append(i)
    
    # Fast memory state
    fast_memory = set(inputs)  # Inputs start in fast memory
    operations = []
    
    # Process vertices in topological order
    for v in topo_order:
        if v in fast_memory:
            continue
        
        if v in inputs:
            # Shouldn't happen, but handle it
            operations.append(['load', v])
            fast_memory.add(v)
            continue
        
        # Ensure all parents are in fast memory
        for p in parents.get(v, []):
            if p not in fast_memory:
                # Need to load p from slow memory
                operations.append(['load', p])
                fast_memory.add(p)
        
        # Compute v
        # Check if we need to evict to make space
        if len(fast_memory) >= fast_memory_budget:
            # Find best vertex to evict
            best_evict = None
            best_next_use = -1
            
            for u in fast_memory:
                if u in outputs:
                    continue
                # Find next use of u after current position
                current_pos = vertex_pos[v]
                next_use = None
                for pos in needed_at.get(u, []):
                    if pos > current_pos:
                        next_use = pos
                        break
                
                if next_use is None:
                    # Never used again, perfect candidate
                    if best_evict is None or next_use > best_next_use:
                        best_evict = u
                        best_next_use = float('inf')
                else:
                    if best_evict is None or next_use > best_next_use:
                        best_evict = u
                        best_next_use = next_use
            
            if best_evict is not None:
                operations.append(['store', best_evict])
                fast_memory.remove(best_evict)
        
        operations.append(['compute', v])
        fast_memory.add(v)
    
    return operations
# EVOLVE-BLOCK-END

