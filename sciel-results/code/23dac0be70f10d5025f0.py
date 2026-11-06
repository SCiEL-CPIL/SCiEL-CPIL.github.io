def pebble(dag, fast_memory_budget):
    from collections import deque
    
    parents = {}
    outputs = set()
    inputs = None
    vertices = None
    
    if isinstance(dag, dict):
        parents = dag.get('parents', {})
        outputs = set(dag.get('outputs', []))
        inputs = dag.get('inputs', None)
        if 'vertices' in dag:
            vertices = list(dag['vertices'])
        elif 'n' in dag:
            vertices = list(range(dag['n']))
        elif 'num_vertices' in dag:
            vertices = list(range(dag['num_vertices']))
    else:
        parents = getattr(dag, 'parents', {})
        outputs = set(getattr(dag, 'outputs', []))
        inputs = getattr(dag, 'inputs', None)
        if hasattr(dag, 'vertices'):
            vertices = list(dag.vertices)
        elif hasattr(dag, 'n'):
            vertices = list(range(dag.n))
    
    if vertices is None:
        vertices = set(parents.keys())
        for p in parents.values():
            vertices.update(p)
        vertices = list(vertices)
    
    if inputs is None:
        all_parents = set()
        for p in parents.values():
            all_parents.update(p)
        inputs = [v for v in vertices if v not in all_parents]
    
    inputs_set = set(inputs)
    
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
    
    if len(topo_order) < len(vertices):
        remaining = [v for v in vertices if v not in set(topo_order)]
        topo_order.extend(remaining)
    
    vertex_index = {v: i for i, v in enumerate(topo_order)}
    
    last_parent_use = {}
    for v in vertices:
        if children[v]:
            last_parent_use[v] = max(vertex_index[c] for c in children[v])
        else:
            last_parent_use[v] = -1
    
    fast_memory = set()
    operations = []
    MAX_OPS = 20000
    
    for v in topo_order:
        if len(operations) >= MAX_OPS:
            break
        is_input = v in inputs_set
        
        if is_input:
            if v not in fast_memory:
                operations.append(['load', v])
                fast_memory.add(v)
        else:
            for p in parents.get(v, []):
                if p not in fast_memory:
                    operations.append(['load', p])
                    fast_memory.add(p)
            operations.append(['compute', v])
            fast_memory.add(v)
        
        while len(fast_memory) > fast_memory_budget and len(operations) < MAX_OPS:
            current_idx = vertex_index[v]
            
            def evict_priority(u):
                lpu = last_parent_use[u]
                if lpu < current_idx:
                    return (0, 0 if u not in outputs else 1)
                else:
                    return (1, lpu, 0 if u not in outputs else 1)
            
            candidates = list(fast_memory)
            candidates.sort(key=evict_priority)
            u = candidates[0]
            
            if u in outputs:
                operations.append(['store', u])
            else:
                if last_parent_use[u] >= current_idx:
                    operations.append(['store', u])
                else:
                    operations.append(['delete', u])
            fast_memory.remove(u)
    
    for u in outputs:
        if u in fast_memory and len(operations) < MAX_OPS:
            operations.append(['store', u])
            fast_memory.remove(u)
    
    for u in list(fast_memory):
        if len(operations) < MAX_OPS:
            operations.append(['delete', u])
            fast_memory.remove(u)
    
    return operations
