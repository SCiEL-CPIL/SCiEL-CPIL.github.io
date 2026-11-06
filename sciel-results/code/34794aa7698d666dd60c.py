def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    deadline = start_time + 80
    
    # Try to understand dag structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        edges = dag.get('edges', [])
        inputs = dag.get('inputs', [])
        outputs = dag.get('outputs', [])
    else:
        vertices = getattr(dag, 'vertices', [])
        edges = getattr(dag, 'edges', [])
        inputs = getattr(dag, 'inputs', [])
        outputs = getattr(dag, 'outputs', [])
    
    # Build adjacency
    parents = {v: [] for v in vertices}
    children = {v: [] for v in vertices}
    for e in edges:
        if len(e) == 2:
            p, c = e
            if c in parents:
                parents[c].append(p)
            if p in children:
                children[p].append(c)
    
    # Topo sort
    from collections import deque
    in_deg = {v: len(parents[v]) for v in vertices}
    q = deque(v for v in vertices if in_deg[v] == 0)
    topo = []
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children.get(v, []):
            in_deg[c] -= 1
            if in_deg[c] == 0:
                q.append(c)
    
    blue = set(inputs)
    ops = []
    
    for v in topo:
        if time.time() > deadline:
            break
        if v in blue:
            continue
        
        # Ensure parents are blue
        for p in parents[v]:
            if p not in blue:
                ops.append(['load', p])
                blue.add(p)
                if len(blue) > fast_memory_budget:
                    # Evict least urgent
                    cands = [x for x in blue if x not in outputs and x not in inputs]
                    if cands:
                        def urg(x):
                            return len([c for c in children.get(x, []) if c not in blue])
                        ev = min(cands, key=urg)
                        ops.append(['store', ev])
                        blue.remove(ev)
        
        if all(p in blue for p in parents[v]):
            ops.append(['compute', v])
            blue.add(v)
            if len(blue) > fast_memory_budget:
                cands = [x for x in blue if x not in outputs and x not in inputs]
                if cands:
                    def urg(x):
                        return len([c for c in children.get(x, []) if c not in blue])
                    ev = min(cands, key=urg)
                    ops.append(['store', ev])
                    blue.remove(ev)
    
    # Ensure outputs blue
    for v in outputs:
        if v not in blue:
            ops.append(['load', v])
            blue.add(v)
            if len(blue) > fast_memory_budget:
                cands = [x for x in blue if x not in outputs and x not in inputs]
                if cands:
                    def urg(x):
                        return len([c for c in children.get(x, []) if c not in blue])
                    ev = min(cands, key=urg)
                    ops.append(['store', ev])
                    blue.remove(ev)
    
    return ops
