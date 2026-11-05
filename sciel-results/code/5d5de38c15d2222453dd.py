import time


def _get(dag, *names, default=None):
    for n in names:
        if isinstance(dag, dict):
            if n in dag:
                return dag[n]
        else:
            if hasattr(dag, n):
                return getattr(dag, n)
    return default


def _parse_dag(dag):
    parents = _get(dag, 'parents', 'parent', default=None)
    outputs = _get(dag, 'outputs', 'output', 'sinks', default=None)
    inputs = _get(dag, 'inputs', 'input', 'sources', default=None)

    if parents is None:
        if isinstance(dag, dict) and len(dag) > 0 and all(
                isinstance(v, (list, tuple, set)) for v in dag.values()):
            parents = {k: list(v) for k, v in dag.items()}
        else:
            parents = {}
    elif isinstance(parents, dict):
        parents = {k: list(v) for k, v in parents.items()}
    elif isinstance(parents, (list, tuple)):
        verts = _get(dag, 'vertices', 'nodes', 'vertex', default=None)
        if verts is None:
            verts = list(range(len(parents)))
        parents = {v: list(p) for v, p in zip(verts, parents)}

    vertices = _get(dag, 'vertices', 'nodes', 'vertex', default=None)
    if vertices is None:
        vs = set()
        for v, ps in parents.items():
            vs.add(v)
            vs.update(ps)
        vertices = sorted(vs, key=lambda x: (str(type(x)), x))
    else:
        vertices = list(vertices)

    if outputs is None:
        dependents = {v: [] for v in vertices}
        for v, ps in parents.items():
            for p in ps:
                if p in dependents:
                    dependents[p].append(v)
        outputs = [v for v in vertices if not dependents.get(v)]
    else:
        outputs = list(outputs)

    if inputs is None:
        inputs = [v for v in vertices if not parents.get(v)]
    else:
        inputs = list(inputs)

    return parents, inputs, outputs, vertices


def _topo_order(parents, vertices):
    from collections import deque
    vset = set(vertices)
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)
    indeg = {v: len([p for p in parents.get(v, []) if p in vset]) for v in vertices}
    q = deque([v for v in vertices if indeg[v] == 0])
    order = []
    indeg2 = dict(indeg)
    while q:
        v = q.popleft()
        order.append(v)
        for c in children.get(v, []):
            indeg2[c] -= 1
            if indeg2[c] == 0:
                q.append(c)
    if len(order) < len(vertices):
        for v in vertices:
            if v not in order:
                order.append(v)
    return order


def pebble(dag, fast_memory_budget):
    start = time.time()
    DEADLINE = 55.0  # safety margin under the 60 s wall-clock kill

    def elapsed():
        return time.time() - start

    def fallback():
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []

    try:
        parents, inputs, outputs, vertices = _parse_dag(dag)
        if not vertices:
            return fallback()
        budget = int(fast_memory_budget)
        if budget < 1:
            budget = 1

        order = _topo_order(parents, vertices)
        order_index = {v: i for i, v in enumerate(order)}
        children = {v: [] for v in vertices}
        for v in vertices:
            for p in parents.get(v, []):
                if p in children:
                    children[p].append(v)
        lu = {}
        for v in vertices:
            if children.get(v):
                lu[v] = max(order_index[c] for c in children[v] if c in order_index)
            else:
                lu[v] = -1

        input_set = set(inputs)
        output_set = set(outputs)

        fast = set(inputs)
        ops = []
        # If inputs alone exceed budget, we are forced to store some (transfers).
        if len(fast) > budget:
            for v in list(fast):
                if len(fast) <= budget:
                    break
                ops.append(['store', v])
                fast.discard(v)

        def need_space():
            return len(fast) >= budget

        def evict_sacrifice():
            # Prefer never-needed values, then values needed latest (safe to delete now).
            cands = [v for v in fast if v not in input_set and v not in output_set]
            if not cands:
                return None

            def key(v):
                l = lu.get(v, -1)
                return 10 ** 9 if l < 0 else l

            cands.sort(key=key)
            return cands[-1]

        def free_space():
            s = evict_sacrifice()
            if s is not None:
                ops.append(['delete', s])
                fast.discard(s)
                return True
            # No free non-input/non-output: store (a transfer) as a last resort.
            for cand in list(fast):
                if cand not in output_set:
                    ops.append(['store', cand])
                    fast.discard(cand)
                    return True
            if fast:
                cand = next(iter(fast))
                ops.append(['store', cand])
                fast.discard(cand)
                return True
            return False

        def ensure_in_fast(v):
            if v in fast:
                return
            if v in input_set:
                # Inputs cannot be recomputed; load them (a transfer) if needed.
                if need_space():
                    free_space()
                ops.append(['load', v])
                fast.add(v)
                return
            # Non-input: bring parents into fast memory, then compute (free).
            for p in parents.get(v, []):
                if p not in fast:
                    ensure_in_fast(p)
            if need_space():
                free_space()
            ops.append(['compute', v])
            fast.add(v)

        for v in order:
            if v in input_set:
                continue
            if elapsed() > DEADLINE:
                break
            ensure_in_fast(v)

        # All outputs must finish blue (in fast memory).
        for o in outputs:
            if o not in fast:
                if elapsed() > DEADLINE:
                    break
                ensure_in_fast(o)

        if len(ops) > 20000:
            ops = ops[:20000]
        return ops
    except Exception:
        return fallback()
