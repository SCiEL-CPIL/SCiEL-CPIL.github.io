def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    DEADLINE = 85.0
    
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        parents = dag.get('parents', {})
        outputs = set(dag.get('outputs', []))
    else:
        vertices = getattr(dag, 'vertices', [])
        parents = getattr(dag, 'parents', {})
        outputs = set(getattr(dag, 'outputs', []))
    
    if not vertices:
        return []
    
    level = {}
    for v in vertices:
        ps = parents.get(v, [])
        if not ps:
            level[v] = 0
        else:
            level[v] = max(level[p] for p in ps) + 1
    
    max_level = max(level.values()) if level else 0
    
    levels = {}
    for v in vertices:
        lv = level[v]
        if lv not in levels:
            levels[lv] = []
        levels[lv].append(v)
    
    in_memory = set()
    for v in vertices:
        if not parents.get(v, []):
            in_memory.add(v)
    
    operations = []
    
    for lv in range(1, max_level + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        current = levels.get(lv, [])
        prev = levels.get(lv - 1, [])
        
        for v in current:
            if time.time() - start_time > DEADLINE:
                break
            operations.append(['compute', v])
            in_memory.add(v)
        
        for v in prev:
            if v not in outputs and v in in_memory:
                operations.append(['delete', v])
                in_memory.discard(v)
    
    for v in outputs:
        if v not in in_memory:
            operations.append(['load', v])
            in_memory.add(v)
    
    return operations
