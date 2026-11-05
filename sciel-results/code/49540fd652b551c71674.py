"""DAG pebbling: minimize total transfers under a fast-memory budget.

Just-in-time pebbling:
  * Topological order; keep a vertex in fast memory only until its last
    consumer is computed, then delete it (freeing a slot) unless it is an
    input (reloadable) or an output (must end blue).
  * When fast memory is full, spill the value whose last use is closest,
    minimizing future recomputation.
  * Recompute a deleted value only if a consumer actually needs it.

This lowers total load/store transfers vs the naive keep-all schedule.
"""
import time
from collections import defaultdict, deque


def pebble(dag, fast_memory_budget):
    start = time.time()
    DEADLINE = 85.0

    # ---------- parse dag defensively ----------
    def _get_parents(v):
        try:
            return list(dag['parents'][v])
        except Exception:
            return []

    def _num_vertices():
        for k in ('n', 'num_vertices', 'num_nodes', 'size'):
            if k in dag:
                try:
                    return int(dag[k])
                except Exception:
                    pass
        try:
            return len(dag['parents'])
        except Exception:
            pass
        try:
            return len(dag['vertices'])
        except Exception:
            return 0

    n = _num_vertices()
    parents = {v: _get_parents(v) for v in range(n)}

    try:
        outputs = set(dag['outputs'])
    except Exception:
        outputs = set(v for v in range(n)
                      if not any(v in parents[u] for u in range(n)))

    children = defaultdict(list)
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)

    inputs = [v for v in range(n) if not parents[v]]

    # topological order
    indeg = {v: len(parents[v]) for v in range(n)}
    q = deque(v for v in range(n) if indeg[v] == 0)
    topo = []
    indeg2 = dict(indeg)
    while q:
        u = q.popleft()
        topo.append(u)
        for c in children[u]:
            indeg2[c] -= 1
            if indeg2[c] == 0:
                q.append(c)
    if len(topo) != n:
        topo = list(range(n))
    pos = {v: i for i, v in enumerate(topo)}

    # last use position (latest consumer); outputs used at the end
    last_use = {}
    for v in range(n):
        lu = -1
        for c in children[v]:
            if pos[c] > lu:
                lu = pos[c]
        if v in outputs:
            lu = max(lu, len(topo) - 1)
        last_use[v] = lu

    # remaining demand (consumers still to compute); outputs need >=1
    demand_left = {}
    for v in range(n):
        d = len(children[v])
        if v in outputs:
            d = max(d, 1)
        demand_left[v] = d

    # ---------- just-in-time schedule ----------
    in_fast = set()
    blue = set(inputs)
    computed = set()
    ops = []

    def spill(k):
        while len(in_fast) >= fast_memory_budget and k > 0:
            best = None
            best_key = None
            for x in in_fast:
                key = (last_use[x], demand_left[x])
                if best_key is None or key < best_key:
                    best_key = key
                    best = x
            if best is None:
                break
            in_fast.remove(best)
            blue.add(best)
            ops.append(['store', best])
            k -= 1

    def ensure_red(v):
        if v in in_fast:
            return
        if v in blue:
            if len(in_fast) >= fast_memory_budget:
                spill(1)
            blue.remove(v)
            in_fast.add(v)
            ops.append(['load', v])
            return
        # deleted -> recompute
        for p in parents[v]:
            ensure_red(p)
        if len(in_fast) >= fast_memory_budget:
            spill(1)
        in_fast.add(v)
        ops.append(['compute', v])
        computed.add(v)

    for v in topo:
        ensure_red(v)
        for p in parents[v]:
            demand_left[p] -= 1
        for p in parents[v]:
            if demand_left[p] <= 0 and p in in_fast:
                in_fast.remove(p)
                if p in inputs:
                    blue.add(p)
        if v in outputs and demand_left[v] <= 0:
            if v in in_fast:
                in_fast.remove(v)
                blue.add(v)
                ops.append(['store', v])

    for v in outputs:
        if v in in_fast:
            in_fast.remove(v)
            blue.add(v)
            ops.append(['store', v])

    # ---------- safety ----------
    if len(ops) > 20000:
        ops = _naive(n, parents, outputs, inputs, topo, fast_memory_budget)

    if time.time() > DEADLINE:
        return ops

    return ops


def _naive(n, parents, outputs, inputs, topo, budget):
    in_fast = set()
    blue = set(inputs)
    ops = []

    def spill():
        cand = [x for x in in_fast if x in inputs] or list(in_fast)
        if not cand:
            return
        x = cand[0]
        in_fast.remove(x)
        blue.add(x)
        ops.append(['store', x])

    for v in topo:
        for p in parents[v]:
            if p not in in_fast:
                if len(in_fast) >= budget:
                    spill()
                if p in blue:
                    blue.remove(p)
                    in_fast.add(p)
                    ops.append(['load', p])
                else:
                    in_fast.add(p)
                    ops.append(['compute', p])
        if len(in_fast) >= budget:
            spill()
        in_fast.add(v)
        ops.append(['compute', v])

    for v in outputs:
        if v in in_fast:
            in_fast.remove(v)
            blue.add(v)
            ops.append(['store', v])
    return ops
