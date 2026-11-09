def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    DEADLINE = 85.0
    
    # Parse DAG
    vertices = []
    parents = {}
    inputs = set()
    outputs = set()
    
    if isinstance(dag, dict):
        vertices = list(dag.get('vertices', dag.get('nodes', [])))
        parents = {k: list(v) for k, v in dag.get('parents', dag.get('edges', {})).items()}
        inputs = set(dag.get('inputs', dag.get('sources', [])))
        outputs = set(dag.get('outputs', dag.get('sinks', [])))
        
        if not inputs:
            all_children = set()
            for ps in parents.values():
                for p in ps:
                    all_children.add(p)
            inputs = set(v for v in vertices if v not in all_children)
        if not outputs:
            all_parents = set()
            for ps in parents.values():
                for p in ps:
                    all_parents.add(p)
            outputs = set(v for v in vertices if v not in all_parents)
    elif isinstance(dag, (list, tuple)):
        if len(dag) > 0 and isinstance(dag[0], (list, tuple)):
            for item in dag:
                v = item[0]
                ps = list(item[1]) if len(item) > 1 and isinstance(item[1], (list, tuple)) else []
                vertices.append(v)
                parents[v] = ps
            all_children = set()
            for ps in parents.values():
                for p in ps:
                    all_children.add(p)
            inputs = set(v for v in vertices if v not in all_children)
            all_parents = set()
            for ps in parents.values():
                for p in ps:
                    all_parents.add(p)
            outputs = set(v for v in vertices if v not in all_parents)
        else:
            vertices = list(dag)
            parents = {v: [] for v in vertices}
            inputs = set(vertices)
            outputs = set(vertices)
    else:
        return []
    
    for v in vertices:
        if v not in parents:
            parents[v] = []
    
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p in children:
                children[p].append(v)
    
    topo_order = list(vertices)
    
    if not vertices:
        return []
    
    fast_mem = set()
    ops = []
    
    pos = {v: i for i, v in enumerate(topo_order)}
    next_use = {}
    for v in topo_order:
        p = pos[v]
        next_idx = len(topo_order)
        for j in range(p + 1, len(topo_order)):
            if v in parents.get(topo_order[j], []):
                next_idx = j
                break
        next_use[v] = next_idx - p
    
    for idx, v in enumerate(topo_order):
        if time.time() - start_time > DEADLINE:
            break
        
        if v in inputs:
            if v not in fast_mem:
                ops.append(['load', v])
                fast_mem.add(v)
        else:
            for p in parents[v]:
                if p not in fast_mem:
                    ops.append(['load', p])
                    fast_mem.add(p)
            ops.append(['compute', v])
            fast_mem.add(v)
        
        while len(fast_mem) > fast_memory_budget:
            if time.time() - start_time > DEADLINE:
                break
            candidates = list(fast_mem)
            def val(x):
                nu = next_use.get(x, 0)
                fo = len(children.get(x, []))
                return fo / (nu + 1)
            candidates.sort(key=val)
            to_del = candidates[0]
            if to_del in fast_mem:
                ops.append(['delete', to_del])
                fast_mem.discard(to_del)
    
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break
        if v in outputs and v in fast_mem:
            ops.append(['store', v])
            fast_mem.discard(v)
    
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
