def pebble(dag, fast_memory_budget):
    ops = []
    budget = int(fast_memory_budget) if fast_memory_budget is not None else 0

    def ga(name):
        if isinstance(dag, dict):
            if name in dag:
                return dag[name]
        else:
            if hasattr(dag, name):
                return getattr(dag, name)
        return None

    def first(*names):
        for n in names:
            v = ga(n)
            if v is not None:
                return v
        return None

    vertices = first('vertices','nodes','V','verts','ids','vertex_list')
    parents_raw = first('parents','pred','preds','in_edges','deps','dependencies')
    children_raw = first('children','succ','succs','out_edges')
    outputs_raw = first('outputs','sinks','out','results','targets','output')
    inputs_raw = first('inputs','sources','src','srcs','input')
    edges_raw = first('edges','E','arcs','links')
    topo_raw = first('topo_order','topological_order','topo','order','linear_extension')

    # normalize vertices
    if vertices is None:
        # try to derive from parents/children/edges
        vertices = []
    vertices = list(vertices) if vertices is not None else []
    vset = set(vertices)

    def norm_map(raw, is_children=False):
        # returns dict vertex -> list of neighbors
        m = {}
        if raw is None:
            return m
        if isinstance(raw, dict):
            for k, val in raw.items():
                m[k] = list(val)
            return m
        if isinstance(raw, (list, tuple)):
            # could be list of lists aligned with vertices, or list of (v, [..])
            if len(raw) == len(vertices) and all(isinstance(x, (list, tuple, set)) for x in raw):
                for i, x in enumerate(raw):
                    m[vertices[i]] = list(x)
            else:
                for item in raw:
                    if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], (list, tuple, set)):
                        m[item[0]] = list(item[1])
                    elif isinstance(item, (list, tuple)) and len(item) == 2:
                        # edge
                        pass
            return m
        return m

    parents_map = norm_map(parents_raw)
    children_map = norm_map(children_raw)

    # derive from edges if needed
    if edges_raw is not None:
        for e in edges_raw:
            if isinstance(e, (list, tuple)) and len(e) == 2:
                u, w = e[0], e[1]
                # assume u -> w, u is parent of w
                parents_map.setdefault(w, [])
                if u not in parents_map[w]:
                    parents_map[w].append(u)
                children_map.setdefault(u, [])
                if w not in children_map[u]:
                    children_map[u].append(w)
                if u not in vset:
                    vertices.append(u); vset.add(u)
                if w not in vset:
                    vertices.append(w); vset.add(w)

    # derive children from parents if missing
    if not children_map and parents_map:
        for w, ps in parents_map.items():
            for u in ps:
                children_map.setdefault(u, [])
                if w not in children_map[u]:
                    children_map[u].append(w)
    # derive parents from children if missing
    if not parents_map and children_map:
        for u, cs in children_map.items():
            for w in cs:
                parents_map.setdefault(w, [])
                if u not in parents_map[w]:
                    parents_map[w].append(u)

    # ensure all vertices present in maps
    for v in vertices:
        parents_map.setdefault(v, [])
        children_map.setdefault(v, [])

    # inputs and outputs
    if inputs_raw is not None:
        inputs = set(list(inputs_raw))
    else:
        inputs = set(v for v in vertices if not parents_map.get(v))
    if outputs_raw is not None:
        outputs = set(list(outputs_raw))
    else:
        outputs = set(v for v in vertices if not children_map.get(v))

    # topological order
    if topo_raw is not None:
        topo = list(topo_raw)
        # ensure all vertices included
        tset = set(topo)
        for v in vertices:
            if v not in tset:
                topo.append(v); tset.add(v)
    else:
        # Kahn
        indeg = {v: len(parents_map.get(v, [])) for v in vertices}
        from collections import deque
        q = deque([v for v in vertices if indeg[v] == 0])
        topo = []
        indeg2 = dict(indeg)
        while q:
            u = q.popleft()
            topo.append(u)
            for w in children_map.get(u, []):
                indeg2[w] = indeg2.get(w, 0) - 1
                if indeg2[w] == 0:
                    q.append(w)
        # append any remaining (cycles) 
        tset = set(topo)
        for v in vertices:
            if v not in tset:
                topo.append(v)

    idx = {v: i for i, v in enumerate(topo)}

    last_use = {}
    for v in vertices:
        ch = children_map.get(v, [])
        if ch:
            last_use[v] = max(idx.get(c, idx[v]) for c in ch)
        else:
            last_use[v] = idx.get(v, 0)

    state = {}
    for v in vertices:
        state[v] = 'blue' if v in inputs else 'absent'
    red = set()

    def ensure_red(v):
        if v not in state:
            state[v] = 'absent'
        if state[v] == 'red':
            return
        if state[v] == 'blue':
            ops.append(['load', v])
            state[v] = 'red'
            red.add(v)
        else:
            # recompute
            for p in parents_map.get(v, []):
                ensure_red(p)
            ops.append(['compute', v])
            state[v] = 'red'
            red.add(v)

    def store_blue(v):
        if state.get(v) == 'red':
            ops.append(['store', v])
            state[v] = 'blue'
            red.discard(v)

    def delete_red(v):
        if state.get(v) == 'red':
            ops.append(['delete', v])
            state[v] = 'absent'
            red.discard(v)

    MAXOPS = 20000
    for i, v in enumerate(topo):
        if len(ops) >= MAXOPS:
            break
        if v in inputs:
            # no compute; but ensure it's available (blue). It is.
            pass
        else:
            for p in parents_map.get(v, []):
                if len(ops) >= MAXOPS: break
                ensure_red(p)
            # free slots to make room for v
            while len(red) >= budget and budget > 0:
                did = False
                for u in list(red):
                    if last_use.get(u, 0) < i:
                        delete_red(u); did = True; break
                if did:
                    continue
                par = set(parents_map.get(v, []))
                cands = [u for u in red if u != v and last_use.get(u,0) >= i]
                nonpar = [u for u in cands if u not in par]
                pool = nonpar if nonpar else cands
                if pool:
                    u = max(pool, key=lambda x: last_use.get(x,0))
                    store_blue(u)
                elif par & set(red):
                    u = next(iter(par & set(red)))
                    store_blue(u)
                elif red:
                    u = next(iter(red))
                    store_blue(u)
                else:
                    break
            if len(ops) >= MAXOPS: break
            ops.append(['compute', v])
            state[v] = 'red'
            red.add(v)
            if v in outputs:
                store_blue(v)
        # delete no-longer-needed red vertices
        for u in list(red):
            if last_use.get(u,0) < i+1 and u not in outputs:
                delete_red(u)
        if len(ops) >= MAXOPS:
            break

    # ensure outputs blue
    for v in outputs:
        if state.get(v) == 'red':
            store_blue(v)
        elif state.get(v) not in ('blue',):
            # absent output: recompute
            ensure_red(v)
            store_blue(v)

    return ops
