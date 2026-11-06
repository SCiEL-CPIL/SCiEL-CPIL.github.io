def pebble(dag, fast_memory_budget):
    vertices = dag.get('vertices', [])
    parents = dag.get('parents', {})
    outputs = set(dag.get('outputs', []))
    inputs = set(dag.get('inputs', []))
    
    # Build children
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Topological sort
    in_degree = {v: 0 for v in vertices}
    for v in vertices:
        for c in children[v]:
            in_degree[c] += 1
    
    from collections import deque
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # For each vertex, compute the list of positions where it is needed (as a parent)
    # A vertex u is needed at position i if u is a parent of topo_order[i]
    needed_at = {v: [] for v in vertices}
    for i, v in enumerate(topo_order):
        for p in parents.get(v, []):
            needed_at[p].append(i)
    
    # For each vertex, the first and last needed position
    first_need = {v: (needed_at[v][0] if needed_at[v] else len(topo_order)) for v in vertices}
    last_need = {v: (needed_at[v][-1] if needed_at[v] else -1) for v in vertices}
    
    # Outputs need to be stored at the end
    # We'll process in topological order
    
    operations = []
    in_fast = set()
    
    # For Belady's algorithm, we need to know the next use of each vertex
    # Let's precompute next_use for each position
    
    # Actually, let's use a simpler approach: when we need to evict, evict the vertex with the farthest next use
    # We can compute next_use on the fly
    
    # For each vertex, maintain a pointer to its next use
    next_use_ptr = {v: 0 for v in vertices}
    
    for i, v in enumerate(topo_order):
        # Get parents of v
        v_parents = parents.get(v, [])
        
        # For each parent, ensure it's in fast memory
        for p in v_parents:
            if p not in in_fast:
                # Need to load p
                if len(in_fast) >= fast_memory_budget:
                    # Evict using Belady's: evict the vertex with the farthest next use
                    # Compute next use for each vertex in in_fast
                    def get_next_use(u):
                        # Find the next position >= i where u is needed
                        ptr = next_use_ptr[u]
                        while ptr < len(needed_at[u]) and needed_at[u][ptr] < i:
                            ptr += 1
                        next_use_ptr[u] = ptr
                        if ptr < len(needed_at[u]):
                            return needed_at[u][ptr]
                        else:
                            return float('inf')  # Never needed again
                    
                    # Find the vertex with the farthest next use (or never needed)
                    best_evict = None
                    best_next = -1
                    for u in in_fast:
                        nu = get_next_use(u)
                        # Prefer evicting vertices that are not outputs
                        if u in outputs:
                            nu = -1  # Don't evict outputs if possible
                        if nu > best_next:
                            best_next = nu
                            best_evict = u
                    
                    if best_evict is not None:
                        operations.append(['delete', best_evict])
                        in_fast.remove(best_evict)
                
                operations.append(['load', p])
                in_fast.add(p)
        
        # Compute v if it's not an input
        if v not in inputs:
            operations.append(['compute', v])
            in_fast.add(v)
    
    # Store outputs
    for v in topo_order:
        if v in outputs and v in in_fast:
            operations.append(['store', v])
            in_fast.remove(v)
    
    # Delete remaining
    for v in list(in_fast):
        operations.append(['delete', v])
        in_fast.remove(v)
    
    return operations
