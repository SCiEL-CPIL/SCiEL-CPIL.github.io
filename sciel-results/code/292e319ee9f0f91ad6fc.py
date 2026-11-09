def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    DEADLINE = 85.0
    
    parents = dag.get('parents', {})
    outputs = set(dag.get('outputs', []))
    
    all_vertices = set(parents.keys()) | outputs
    for ps in parents.values():
        all_vertices.update(ps)
    
    children = {v: [] for v in all_vertices}
    for v, ps in parents.items():
        for p in ps:
            if p in all_vertices:
                children[p].append(v)
    
    inputs = [v for v in all_vertices if not parents.get(v, [])]
    
    from collections import deque
    in_degree = {v: len([p for p in parents.get(v, []) if p in all_vertices]) for v in all_vertices}
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    state = {}
    for v in all_vertices:
        state[v] = 'blue' if v in inputs else 'gone'
    
    fast_mem = set(v for v in all_vertices if state[v] == 'blue')
    rem_children = {v: len(children.get(v, [])) for v in all_vertices}
    
    ops = []
    
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break
        
        if v in inputs:
            for c in children.get(v, []):
                rem_children[c] -= 1
            continue
        
        for p in parents.get(v, []):
            if p in all_vertices and state[p] == 'red':
                ops.append(['load', p])
                state[p] = 'blue'
                fast_mem.add(p)
        
        ops.append(['compute', v])
        state[v] = 'blue'
        fast_mem.add(v)
        
        for p in parents.get(v, []):
            if p in all_vertices:
                rem_children[p] -= 1
        
        for p in parents.get(v, []):
            if p in all_vertices and rem_children[p] == 0 and state[p] == 'blue' and p not in outputs:
                ops.append(['delete', p])
                state[p] = 'gone'
                fast_mem.discard(p)
        
        while len(fast_mem) > fast_memory_budget:
            candidates = [x for x in fast_mem if x not in outputs]
            if not candidates:
                break
            candidates.sort(key=lambda x: rem_children[x])
            x = candidates[0]
            ops.append(['store', x])
            state[x] = 'red'
            fast_mem.discard(x)
    
    for o in outputs:
        if o in all_vertices and state[o] == 'red':
            ops.append(['load', o])
            state[o] = 'blue'
            fast_mem.add(o)
    
    return ops
