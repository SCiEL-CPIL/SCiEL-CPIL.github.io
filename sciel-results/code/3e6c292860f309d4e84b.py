def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    dag: dict with keys:
      - 'vertices': list of vertex IDs (topologically ordered)
      - 'parents': dict mapping vertex -> list of parent vertices
      - 'inputs': list of input vertex IDs
      - 'outputs': list of output vertex IDs
    
    fast_memory_budget: int, number of slots in fast memory
    
    Returns: list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    import time
    start_time = time.time()
    DEADLINE = 85  # 5s margin before 90s kill
    
    vertices = dag['vertices']
    parents = dag['parents']
    inputs = set(dag['inputs'])
    outputs = set(dag['outputs'])
    
    n = len(vertices)
    budget = fast_memory_budget
    
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents[v]:
            children[p].append(v)
    
    # Topological order is given by vertices list
    
    # Track state: 'blue' (in fast memory), 'red' (in slow memory), 'uncomputed'
    state = {}
    for v in vertices:
        if v in inputs:
            state[v] = 'blue'  # inputs start blue
        else:
            state[v] = 'uncomputed'
    
    # Fast memory contents
    fast_mem = set()
    for v in inputs:
        fast_mem.add(v)
    
    operations = []
    
    def transfer_count():
        return sum(1 for op, v in operations if op in ('load', 'store'))
    
    # If budget is sufficient to hold all vertices, just compute everything
    if budget >= n:
        for v in vertices:
            if v in inputs:
                continue
            # Ensure parents are in fast memory
            for p in parents[v]:
                if state[p] == 'red':
                    operations.append(['load', p])
                    state[p] = 'blue'
                    fast_mem.add(p)
            operations.append(['compute', v])
            state[v] = 'blue'
            fast_mem.add(v)
        # Store non-outputs
        for v in vertices:
            if v not in outputs and state[v] == 'blue':
                operations.append(['store', v])
                state[v] = 'red'
                fast_mem.remove(v)
        return operations
    
    # For limited budget, use a greedy strategy
    # Process in topological order, evict when necessary
    
    # Determine when each vertex is last needed
    last_use = {}
    for i, v in enumerate(vertices):
        # v is needed when it's computed, and when any child is computed
        last_needed = i
        for c in children[v]:
            ci = vertices.index(c)
            last_needed = max(last_needed, ci)
        last_use[v] = last_needed
    
    # Greedy: process in topological order
    # When fast memory is full, evict the vertex with the furthest last use
    
    for i, v in enumerate(vertices):
        if time.time() - start_time > DEADLINE:
            break
        
        if v in inputs:
            continue
        
        # Ensure parents are in fast memory
        for p in parents[v]:
            if state[p] == 'uncomputed':
                # This shouldn't happen if we process in topological order
                pass
            elif state[p] == 'red':
                operations.append(['load', p])
                state[p] = 'blue'
                fast_mem.add(p)
        
        # If fast memory is full, evict something
        while len(fast_mem) >= budget:
            # Find vertex in fast_mem with furthest last use (not an output, not current vertex's parent)
            candidates = [x for x in fast_mem if x != v and x not in [p for p in parents[v]]]
            if not candidates:
                # Can't evict, this is a problem
                break
            # Evict the one with furthest last use
            to_evict = max(candidates, key=lambda x: last_use[x])
            operations.append(['store', to_evict])
            state[to_evict] = 'red'
            fast_mem.remove(to_evict)
        
        # Compute v
        operations.append(['compute', v])
        state[v] = 'blue'
        fast_mem.add(v)
    
    # Ensure outputs are blue
    for v in outputs:
        if state[v] == 'red':
            operations.append(['load', v])
            state[v] = 'blue'
            fast_mem.add(v)
        elif state[v] == 'uncomputed':
            # Need to compute it
            for p in parents[v]:
                if state[p] == 'red':
                    operations.append(['load', p])
                    state[p] = 'blue'
                    fast_mem.add(p)
            # Evict if needed
            while len(fast_mem) >= budget:
                candidates = [x for x in fast_mem if x != v and x not in outputs]
                if not candidates:
                    break
                to_evict = max(candidates, key=lambda x: last_use[x])
                operations.append(['store', to_evict])
                state[to_evict] = 'red'
                fast_mem.remove(to_evict)
            operations.append(['compute', v])
            state[v] = 'blue'
            fast_mem.add(v)
    
    # Check operation limit
    if len(operations) > 20000:
        # Truncate - this is a fallback
        operations = operations[:20000]
    
    return operations
