def pebble(dag, fast_memory_budget):
    parents = dag['parents']
    outputs = set(dag.get('outputs', []))
    inputs = set(dag.get('inputs', []))
    
    # Get all vertices
    all_vertices = set(inputs) | outputs
    for v, ps in parents.items():
        all_vertices.add(v)
        for p in ps:
            all_vertices.add(p)
    
    # Topological sort
    in_degree = {v: 0 for v in all_vertices}
    children_map = {v: [] for v in all_vertices}
    for v, ps in parents.items():
        in_degree[v] = len(ps)
        for p in ps:
            children_map[p].append(v)
    
    from collections import deque
    queue = deque()
    for v in all_vertices:
        if in_degree[v] == 0:
            queue.append(v)
    
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children_map[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # State tracking
    red = set()
    blue = set(inputs)
    
    operations = []
    
    for v in topo_order:
        if v in inputs:
            continue
        
        # Ensure all parents are red
        for p in parents[v]:
            if p not in red:
                if p in blue:
                    red.add(p)
                    blue.discard(p)
                    operations.append(['load', p])
                else:
                    red.add(p)
                    operations.append(['load', p])
        
        # Free memory if needed
        if len(red) + 1 > fast_memory_budget:
            for rv in sorted(list(red), key=lambda x: topo_order.index(x) if x in topo_order else 0):
                if len(red) + 1 <= fast_memory_budget:
                    break
                if rv not in outputs:
                    red.discard(rv)
                    operations.append(['delete', rv])
        
        # Compute
        red.add(v)
        operations.append(['compute', v])
    
    # Store all outputs
    for v in outputs:
        if v in red:
            red.discard(v)
            blue.add(v)
            operations.append(['store', v])
    
    return operations
