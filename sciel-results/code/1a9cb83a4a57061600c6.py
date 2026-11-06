def pebble(dag, fast_memory_budget):
    import time
    start = time.time()
    DEADLINE = 55.0
    
    # Extract DAG structure
    parents = None
    outputs = None
    n = None
    
    if hasattr(dag, 'parents'):
        parents = dag.parents
    elif isinstance(dag, dict) and 'parents' in dag:
        parents = dag['parents']
    
    if parents is None:
        return []
    
    n = len(parents)
    
    if hasattr(dag, 'outputs'):
        outputs = list(dag.outputs)
    elif isinstance(dag, dict) and 'outputs' in dag:
        outputs = list(dag['outputs'])
    else:
        # Outputs are vertices with no children
        children = [[] for _ in range(n)]
        for i in range(n):
            for p in parents[i]:
                children[p].append(i)
        outputs = [i for i in range(n) if len(children[i]) == 0]
    
    inputs = [i for i in range(n) if len(parents[i]) == 0]
    
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in parents[i]:
            children[p].append(i)
    
    # Layer computation
    layer = [0] * n
    for i in range(n):
        if parents[i]:
            layer[i] = max(layer[p] + 1 for p in parents[i])
    
    # Greedy pebbling
    in_mem = set()
    ops = []
    
    # ref_count: how many more children of i need to be processed
    ref = [len(children[i]) for i in range(n)]
    
    # Load inputs
    for i in inputs:
        ops.append(['load', i])
        in_mem.add(i)
    
    # Process in topological order
    for v in range(n):
        # Ensure v is in memory
        if v not in in_mem:
            # Load parents if needed
            for p in parents[v]:
                if p not in in_mem:
                    ops.append(['load', p])
                    in_mem.add(p)
            ops.append(['compute', v])
            in_mem.add(v)
        
        # Free up space if needed
        while len(in_mem) > fast_memory_budget:
            # Find best candidate to store
            # Prefer: ref==0, not output, lowest layer
            cands = [x for x in in_mem if x not in outputs]
            if not cands:
                break
            # Sort by (ref_count, layer)
            cands.sort(key=lambda x: (ref[x], layer[x]))
            to_store = cands[0]
            ops.append(['store', to_store])
            in_mem.discard(to_store)
        
        # Decrement ref counts of parents
        for p in parents[v]:
            ref[p] -= 1
    
    # Ensure outputs are in memory
    for o in outputs:
        if o not in in_mem:
            ops.append(['load', o])
            in_mem.add(o)
    
    if time.time() - start > DEADLINE:
        pass
    
    return ops
