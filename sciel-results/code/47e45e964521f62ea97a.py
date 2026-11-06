def pebble(dag, fast_memory_budget):
    from collections import deque
    
    # Extract DAG structure
    if isinstance(dag, dict):
        parents = dag.get('parents', {})
        children = dag.get('children', {})
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
    else:
        raise ValueError("Unknown dag format")
    
    # Build children if not provided
    if not children:
        children = {}
        for n, ps in parents.items():
            for p in ps:
                children.setdefault(p, []).append(n)
    
    # All nodes
    all_nodes = set(parents.keys()) | set(children.keys()) | inputs | outputs
    
    # Topological order
    in_degree = {n: len(parents.get(n, [])) for n in all_nodes}
    q = deque()
    for n, d in in_degree.items():
        if d == 0:
            q.append(n)
    
    topo = []
    while q:
        n = q.popleft()
        topo.append(n)
        for c in children.get(n, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                q.append(c)
    
    # remaining_children
    remaining_children = {n: len(children.get(n, [])) for n in all_nodes}
    
    # State
    state = {}
    for n in all_nodes:
        if n in inputs:
            state[n] = 'blue'
        else:
            state[n] = 'gone'
    
    fast_memory = set()
    operations = []
    
    for n in topo:
        # Ensure parents are red
        for p in parents.get(n, []):
            if state[p] == 'blue':
                operations.append(['load', p])
                state[p] = 'red'
                fast_memory.add(p)
            elif state[p] == 'gone':
                # Need to recompute p - complex, skip for now or handle
                # For safety, this shouldn't happen with our strategy
                pass
        
        # If input, load
        if n in inputs:
            if state[n] == 'blue':
                operations.append(['load', n])
                state[n] = 'red'
                fast_memory.add(n)
        else:
            # Make room if needed
            if len(fast_memory) >= fast_memory_budget:
                # Find victim
                candidates = [x for x in fast_memory if x != n]
                safe = [x for x in candidates if remaining_children.get(x, 0) == 0 and x not in outputs]
                if safe:
                    victim = safe[0]
                    operations.append(['delete', victim])
                    state[victim] = 'gone'
                    fast_memory.remove(victim)
                else:
                    # Store a node to free fast memory
                    victim = min(candidates, key=lambda x: remaining_children.get(x, 0))
                    operations.append(['store', victim])
                    state[victim] = 'blue'
                    fast_memory.remove(victim)
            
            operations.append(['compute', n])
            state[n] = 'red'
            fast_memory.add(n)
        
        # Update remaining_children
        for p in parents.get(n, []):
            remaining_children[p] -= 1
        
        # If output, store
        if n in outputs:
            operations.append(['store', n])
            state[n] = 'blue'
            fast_memory.discard(n)
    
    return operations
