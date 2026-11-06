def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    DEADLINE = 85.0
    
    # Try to parse the DAG structure
    # Common formats:
    # 1. dag = {'vertices': [...], 'parents': {v: [...]}, 'inputs': [...], 'outputs': [...]}
    # 2. dag = {'n': n, 'parents': [...], 'inputs': [...], 'outputs': [...]}
    # 3. dag is a dict with various keys
    
    # Extract structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', None)
        parents = dag.get('parents', {})
        inputs = dag.get('inputs', [])
        outputs = dag.get('outputs', [])
        
        if vertices is None:
            # Infer from parents
            all_v = set()
            for v, ps in parents.items():
                all_v.add(v)
                all_v.update(ps)
            vertices = sorted(all_v) if all_v else []
        
        # Ensure parents is a dict
        if not isinstance(parents, dict):
            # Maybe it's a list of lists?
            if isinstance(parents, list):
                parents_dict = {}
                for i, ps in enumerate(parents):
                    parents_dict[i] = ps
                parents = parents_dict
                if vertices is None:
                    vertices = list(range(len(parents)))
    
    # Handle case where parents might be missing for some vertices
    for v in vertices:
        if v not in parents:
            parents[v] = []
    
    inputs = set(inputs)
    outputs = set(outputs)
    
    # Build children map
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p in children:
                children[p].append(v)
    
    # Topological sort
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    from collections import deque
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        for v in vertices:
            if v not in set(topo_order):
                topo_order.append(v)
    
    # Compute layers
    layer = {}
    for v in topo_order:
        if v in inputs:
            layer[v] = 0
        else:
            ps = parents.get(v, [])
            if ps:
                layer[v] = max(layer.get(p, 0) for p in ps) + 1
            else:
                layer[v] = 0
    
    max_layer = max(layer.values()) if layer else 0
    
    # Compute last use
    last_use = {}
    for v in vertices:
        if children.get(v):
            max_child_layer = max(layer.get(c, 0) for c in children[v])
            last_use[v] = max_child_layer
        else:
            if v in outputs:
                last_use[v] = max_layer
            else:
                last_use[v] = layer.get(v, 0)
    
    # Pebbling strategy
    in_fast = set(inputs)
    operations = []
    
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break
        
        if v in inputs:
            continue
        
        # Ensure parents are in fast memory
        for p in parents.get(v, []):
            if p not in in_fast:
                operations.append(['load', p])
                in_fast.add(p)
        
        # Check space
        if len(in_fast) + 1 > fast_memory_budget:
            # Evict
            evict_candidates = []
            for x in in_fast:
                if x in parents.get(v, []):
                    continue
                lu = last_use.get(x, 0)
                evict_candidates.append((lu, x))
            
            evict_candidates.sort(reverse=True)
            while len(in_fast) + 1 > fast_memory_budget and evict_candidates:
                evict_v = evict_candidates.pop()[1]
                operations.append(['store', evict_v])
                in_fast.remove(evict_v)
        
        operations.append(['compute', v])
        in_fast.add(v)
    
    # Ensure outputs are in fast memory
    for v in outputs:
        if v not in in_fast:
            operations.append(['load', v])
            in_fast.add(v)
    
    return operations
