def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    deadline = start_time + 82

    # Fallback to baseline
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        baseline_ops = pebble_baseline(dag, fast_memory_budget)
    except Exception:
        baseline_ops = None

    def _extract(dag):
        if isinstance(dag, dict):
            parents_dict = dict(dag.get('parents', {}))
            inputs = list(dag.get('inputs', []))
            outputs = set(dag.get('outputs', []))
        else:
            parents_dict = dict(getattr(dag, 'parents', {}))
            inputs = list(getattr(dag, 'inputs', []))
            outputs = set(getattr(dag, 'outputs', []))
        return parents_dict, inputs, outputs

    def _topo_order(parents_dict, inputs):
        all_vertices = set(inputs)
        for v, ps in parents_dict.items():
            all_vertices.add(v)
            all_vertices.update(ps)
        in_deg = {v: len(parents_dict.get(v, [])) for v in all_vertices}
        children = {v: [] for v in all_vertices}
        for v, ps in parents_dict.items():
            for p in ps:
                children[p].append(v)
        from collections import deque
        q = deque([v for v in all_vertices if in_deg[v] == 0])
        order = []
        while q:
            v = q.popleft()
            order.append(v)
            for c in children[v]:
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    q.append(c)
        return order, children, all_vertices

    def _transfers(ops):
        return sum(1 for op in ops if op[0] in ('load', 'store'))

    def _simulate(parents_dict, inputs, outputs, topo_order, children,
                  keep_prob, fast_memory_budget, deadline):
        in_fast = set()
        ops = []
        # LRU tracking: last access time per vertex
        last_access = {}
        t = 0
        for v in topo_order:
            if time.time() > deadline - 3:
                break
            if v in inputs:
                if v not in in_fast:
                    if len(in_fast) >= fast_memory_budget:
                        # evict: prefer no-future, then lowest keep_prob, then oldest
                        cands = []
                        for u in list(in_fast):
                            if u in outputs:
                                continue
                            fu = 0
                            for c in children.get(u, []):
                                if c not in in_fast and topo_order.index(c) > topo_order.index(v):
                                    fu += 1
                            cands.append((fu, keep_prob.get(u, 0.5), last_access.get(u, -1), u))
                        cands.sort(key=lambda x: (x[0], x[1], x[2], str(x[3])))
                        if cands:
                            u = cands[0][3]
                            in_fast.discard(u)
                            ops.append(['delete', u])
                    in_fast.add(v)
                    ops.append(['load', v])
                last_access[v] = t
            else:
                ps = parents_dict.get(v, [])
                for p in ps:
                    if p not in in_fast:
                        if len(in_fast) >= fast_memory_budget:
                            cands = []
                            for u in list(in_fast):
                                if u in outputs:
                                    continue
                                fu = 0
                                for c in children.get(u, []):
                                    if c not in in_fast and topo_order.index(c) > topo_order.index(v):
                                        fu += 1
                                cands.append((fu, keep_prob.get(u, 0.5), last_access.get(u, -1), u))
                            cands.sort(key=lambda x: (x[0], x[1], x[2], str(x[3])))
                            if cands:
                                u = cands[0][3]
                                in_fast.discard(u)
                                ops.append(['delete', u])
                        in_fast.add(p)
                        ops.append(['load', p])
                        last_access[p] = t
                in_fast.add(v)
                ops.append(['compute', v])
                last_access[v] = t
            t += 1
            # immediate cleanup
            if future_uses.get(v, 0) == 0 and v not in outputs:
                if v in in_fast:
                    in_fast.discard(v)
                    ops.append(['delete', v])
            elif v in outputs:
                if v in in_fast:
                    in_fast.discard(v)
                    ops.append(['store', v])
        # final store outputs
        for v in outputs:
            if v in in_fast:
                in_fast.discard(v)
                ops.append(['store', v])
        return ops

    try:
        parents_dict, inputs, outputs = _extract(dag)
        topo_order, children, all_vertices = _topo_order(parents_dict, inputs)

        # future uses count (static)
        future_uses = {v: len(children.get(v, [])) for v in all_vertices}
        # layer
        layer = {}
        for v in topo_order:
            if v in inputs:
                layer[v] = 0
            else:
                ps = parents_dict.get(v, [])
                layer[v] = (max(layer[p] for p in ps) + 1) if ps else 0
        max_layer = max(layer.values()) if layer else 1

        import random
        best_ops = None
        best_transfers = float('inf')

        # Multiple randomized keep-probability strategies
        for seed in range(12):
            if time.time() > deadline - 5:
                break
            random.seed(seed)
            keep_prob = {}
            for v in topo_order:
                if v in inputs:
                    keep_prob[v] = 1.0
                else:
                    use_factor = min(1.0, future_uses.get(v, 0) / 3.0)
                    layer_factor = 1.0 - (layer[v] / (max_layer + 1)) * 0.4
                    keep_prob[v] = max(0.15, use_factor * layer_factor + random.uniform(0, 0.25))
            ops = _simulate(parents_dict, inputs, outputs, topo_order, children,
                            keep_prob, fast_memory_budget, deadline)
            tr = _transfers(ops)
            if tr < best_transfers:
                best_transfers = tr
                best_ops = ops

        # Local-search polish: try swapping eviction order by perturbing keep_prob
        if best_ops is not None and time.time() < deadline - 8:
            for _ in range(6):
                if time.time() > deadline - 3:
                    break
                random.seed(random.randint(0, 10**6))
                keep_prob = {}
                for v in topo_order:
                    if v in inputs:
                        keep_prob[v] = 1.0
                    else:
                        use_factor = min(1.0, future_uses.get(v, 0) / 3.0)
                        layer_factor = 1.0 - (layer[v] / (max_layer + 1)) * 0.4
                        base = max(0.15, use_factor * layer_factor)
                        keep_prob[v] = max(0.1, min(1.0, base + random.gauss(0, 0.15)))
                ops = _simulate(parents_dict, inputs, outputs, topo_order, children,
                                keep_prob, fast_memory_budget, deadline)
                tr = _transfers(ops)
                if tr < best_transfers:
                    best_transfers = tr
                    best_ops = ops

        if baseline_ops is not None:
            bt = _transfers(baseline_ops)
            if best_ops is not None and best_transfers < bt:
                return best_ops
            return baseline_ops
        else:
            return best_ops if best_ops is not None else []

    except Exception:
        if baseline_ops is not None:
            return baseline_ops
        return []
