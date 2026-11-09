def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    DEADLINE = 85.0
    
    vertices = dag.get('vertices', [])
    parents = dag.get('parents', {})
    outputs = dag.get('outputs', [])
    
    if not vertices:
        if 'nodes' in dag:
            vertices = list(dag['nodes'])
        if 'edges' in dag:
            parents = {v: [] for v in vertices}
            for edge in dag['edges']:
                u, v = edge[0], edge[1]
                if v not in parents:
                    parents[v] = []
                parents[v].append(u)
        if 'output_vertices' in dag:
            outputs = list(dag['output_vertices'])
    
    if not vertices:
        return []
    
    if not parents:
        parents = {v: [] for v in vertices}
    
    if not outputs:
        children = {v: [] for v in vertices}
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
        outputs = [v for v in vertices if not children[v]]
    
    # Topological sort
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    from collections import deque
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for u in vertices:
            if v in parents.get(u, []):
                in_degree[u] -= 1
                if in_degree[u] == 0:
                    queue.append(u)
    
    if len(topo_order) != len(vertices):
        topo_order = list(vertices)
    
    input_vertices = [v for v in vertices if len(parents.get(v, [])) == 0]
    
    # Build reverse dependency: which vertices depend on each vertex
    dependents = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            dependents[p].append(v)
    
    operations = []
    fast_mem = set()
    
    # Inputs start blue - assume they're in fast memory
    for v in input_vertices:
        fast_mem.add(v)
    
    # Track which vertices have been computed
    computed = set()
    
    for v in topo_order:
        if v in input_vertices:
            computed.add(v)
            continue
        
        parent_list = parents.get(v, [])
        
        # Load parents not in fast memory
        for p in parent_list:
            if p not in fast_mem:
                operations.append(['load', p])
                fast_mem.add(p)
        
        # Compute
        operations.append(['compute', v])
        fast_mem.add(v)
        computed.add(v)
        
        # Manage fast memory: evict if over budget
        while len(fast_mem) > fast_memory_budget:
            # Find a value to evict: prefer non-output, non-needed
            best_evict = None
            for u in list(fast_mem):
                if u in outputs:
                    continue
                # Check if u is needed by any uncomputed vertex
                needed = False
                for w in dependents.get(u, []):
                    if w not in computed:
                        needed = True
                        break
                if not needed:
                    best_evict = u
                    break
            if best_evict is None:
                # Evict any non-output
                for u in list(fast_mem):
                    if u not in outputs:
                        best_evict = u
                        break
            if best_evict is None:
                # Evict any
                best_evict = next(iter(fast_mem))
            
            operations.append(['store', best_evict])
            fast_mem.discard(best_evict)
        
        if time.time() - start_time > DEADLINE:
            break
    
    # Ensure outputs are in fast memory at end
    for v in outputs:
        if v not in fast_mem:
            # Need to reload
            # First, make room if needed
            while len(fast_mem) >= fast_memory_budget:
                # Evict non-output
                evicted = False
                for u in list(fast_mem):
                    if u not in outputs:
                        operations.append(['store', u])
                        fast_mem.discard(u)
                        evicted = True
                        break
                if not evicted:
                    break
            operations.append(['load', v])
            fast_mem.add(v)
    
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
