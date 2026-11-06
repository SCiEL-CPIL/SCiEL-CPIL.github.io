def pebble(dag, fast_memory_budget):
    # Extract structure
    if hasattr(dag, 'vertices'):
        vertices = list(dag.vertices)
    elif hasattr(dag, 'parents') and dag.parents:
        vertices = list(range(max(dag.parents.keys()) + 1))
    else:
        vertices = []
    
    parents = {}
    if hasattr(dag, 'parents'):
        for v, p in dag.parents.items():
            parents[v] = list(p)
    for v in vertices:
        if v not in parents:
            parents[v] = []
    
    outputs = set()
    if hasattr(dag, 'outputs'):
        outputs = set(dag.outputs)
    
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents[v]:
            children[p].append(v)
    
    in_degree = {v: len(parents[v]) for v in vertices}
    from collections import deque
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo = []
    in_deg = dict(in_degree)
    while queue:
        v = queue.popleft()
        topo.append(v)
        for c in children[v]:
            in_deg[c] -= 1
            if in_deg[c] == 0:
                queue.append(c)
    
    inputs = [v for v in vertices if in_degree[v] == 0]
    
    in_memory = set(inputs)
    ops = []
    remaining_children = {v: len(children[v]) for v in vertices}
    
    for v in topo:
        if v not in in_memory:
            ops.append(('compute', v))
            in_memory.add(v)
        
        for p in parents[v]:
            remaining_children[p] -= 1
        
        while len(in_memory) > fast_memory_budget:
            candidates = [u for u in in_memory if remaining_children[u] == 0 and u not in outputs]
            if not candidates:
                candidates = [u for u in in_memory if u not in outputs]
                if not candidates:
                    break
                candidates = sorted(candidates, key=lambda u: remaining_children[u])
            to_delete = candidates[0]
            ops.append(('delete', to_delete))
            in_memory.remove(to_delete)
    
    return ops
