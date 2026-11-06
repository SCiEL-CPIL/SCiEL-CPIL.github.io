def pebble(dag, fast_memory_budget):
    # Extract structure
    if isinstance(dag, dict):
        vertices = list(dag.get('vertices', []))
        parents = dag.get('parents', {})
        children = dag.get('children', {})
        outputs = set(dag.get('outputs', []))
        inputs = set(dag.get('inputs', []))
        topo = dag.get('topo_order', [])
    else:
        # Assume dag has attributes
        vertices = list(dag.vertices)
        parents = dag.parents
        children = dag.children
        outputs = set(dag.outputs)
        inputs = set(dag.inputs)
        topo = dag.topo_order
    
    # If no topo order, compute it
    if not topo:
        # Topological sort
        in_degree = {v: len(parents[v]) for v in vertices}
        from collections import deque
        queue = deque([v for v in vertices if in_degree[v] == 0])
        topo = []
        while queue:
            v = queue.popleft()
            topo.append(v)
            for c in children[v]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
    
    in_ram = set(inputs)
    on_disk = set()
    computed = set(inputs)
    ops = []
    
    for v in topo:
        if v in in_ram:
            continue
        
        # Ensure parents are in RAM
        for p in parents[v]:
            if p not in in_ram:
                if p in on_disk:
                    ops.append(['load', p])
                    in_ram.add(p)
                    on_disk.discard(p)
        
        # Compute v
        ops.append(['compute', v])
        in_ram.add(v)
        computed.add(v)
        
        # If RAM is full, evict
        while len(in_ram) > fast_memory_budget:
            candidates = [x for x in in_ram if x not in outputs]
            if not candidates:
                break
            
            # Prefer evicting vertices with few uncomputed children
            def remaining_children(x):
                return sum(1 for c in children[x] if c not in computed)
            
            candidates.sort(key=remaining_children)
            evict = candidates[0]
            ops.append(['store', evict])
            in_ram.discard(evict)
            on_disk.add(evict)
    
    # Ensure outputs are in RAM
    for v in outputs:
        if v not in in_ram:
            if v in on_disk:
                ops.append(['load', v])
                in_ram.add(v)
                on_disk.discard(v)
    
    return ops
