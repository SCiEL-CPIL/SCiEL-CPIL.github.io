def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling with demand-based eviction.
    """
    # Extract DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        parents = dag.get('parents', {})
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
        topo_order = dag.get('topo_order', [])
    else:
        vertices = getattr(dag, 'vertices', [])
        parents = getattr(dag, 'parents', {})
        inputs = set(getattr(dag, 'inputs', []))
        outputs = set(getattr(dag, 'outputs', []))
        topo_order = getattr(dag, 'topo_order', [])
    
    if not topo_order:
        from collections import deque
        in_degree = {v: 0 for v in vertices}
        children_map = {v: [] for v in vertices}
        for v in vertices:
            for p in parents.get(v, []):
                in_degree[v] += 1
                if p in children_map:
                    children_map[p].append(v)
        queue = deque([v for v in vertices if in_degree[v] == 0])
        topo_order = []
        while queue:
            v = queue.popleft()
            topo_order.append(v)
            for c in children_map.get(v, []):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
    
    # Count children for each vertex (future demand)
    children_count = {v: 0 for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children_count:
                children_count[p] += 1
    
    fast_memory = set()
    stored = set()
    operations = []
    
    # Inputs start in fast memory
    for v in inputs:
        fast_memory.add(v)
    
    for v in topo_order:
        if v in inputs:
            continue
        
        # Ensure parents are in fast memory
        for p in parents.get(v, []):
            if p not in fast_memory:
                operations.append(['load', p])
                fast_memory.add(p)
        
        # Compute
        operations.append(['compute', v])
        fast_memory.add(v)
        
        # Store only if it will be needed again (has children)
        if children_count.get(v, 0) > 0:
            operations.append(['store', v])
            stored.add(v)
        
        # Evict if over budget
        while len(fast_memory) > fast_memory_budget:
            # Evict vertex with lowest remaining demand (excluding inputs and current vertex)
            candidates = [u for u in fast_memory if u not in inputs and u != v]
            if not candidates:
                candidates = [u for u in fast_memory if u not in inputs]
            if not candidates:
                candidates = list(fast_memory)
            
            # Pick the one with lowest children_count
            best = min(candidates, key=lambda u: children_count.get(u, 0))
            operations.append(['delete', best])
            fast_memory.discard(best)
    
    # Ensure outputs are in fast memory
    for v in outputs:
        if v not in fast_memory:
            operations.append(['load', v])
            fast_memory.add(v)
    
    return operations
