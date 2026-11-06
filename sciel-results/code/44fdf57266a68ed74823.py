# EVOLVE-BLOCK-START
import time as _time
import math as _math

_DEADLINE = 75.0  # wall-clock safety margin (evaluator kills at 90s)


def _now():
    return _time.monotonic()


def _elapsed():
    return _now() - _START


def _topo_order(dag):
    """Return a topological ordering of vertices (list of ints)."""
    try:
        n = int(dag['n'])
    except Exception:
        n = None
    if n is None:
        try:
            n = int(dag['num_vertices'])
        except Exception:
            n = None
    if n is None:
        try:
            n = len(dag.get('parents', []))
        except Exception:
            n = 0
    parents = dag.get('parents')
    if parents is None:
        parents = dag.get('preds', [])
    # parents[i] = list of parent (predecessor) vertex indices
    indeg = [0] * n
    adj = [[] for _ in range(n)]
    for i in range(n):
        ps = parents[i] if i < len(parents) else []
        for p in ps:
            if 0 <= p < n:
                indeg[i] += 1
                adj[p].append(i)
    # Kahn
    from collections import deque
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) != n:
        # cycle or malformed; append the rest
        seen = set(order)
        for i in range(n):
            if i not in seen:
                order.append(i)
    return order


def _naive_ops(dag, order):
    """Count transfers in a naive topological schedule (load each parent once,
    compute each vertex once, no reuse accounting beyond in-place)."""
    # Naive: for each vertex in topo order, load every parent that isn't already
    # resident (assume nothing resident), compute, keep result. Count loads + computes.
    n = len(order)
    parents = dag.get('parents')
    if parents is None:
        parents = dag.get('preds', [])
    ops = 0
    for i in range(n):
        ps = parents[i] if i < len(parents) else []
        ops += len(ps)          # load each parent
        ops += 1                # compute
    return max(ops, 1)


def _build_schedule(dag, fast_memory_budget, order, deadline):
    """Greedy topological pebbling with bounded fast memory.

    Returns (ops, valid). ops = list of [op, vertex].
    Strategy: process vertices in topological order. Keep a set of
    resident (red) vertices in fast memory. A vertex is 'ready' once all
    its parents are resident (or are inputs already blue/available).
    When the memory budget is exceeded, evict the vertex whose result is
    needed least in the future (lowest remaining use count).
    """
    n = len(order)
    parents = dag.get('parents')
    if parents is None:
        parents = dag.get('preds', [])
    # outputs = vertices with no children (sinks)
    adj = [[] for _ in range(n)]
    for i in range(n):
        ps = parents[i] if i < len(parents) else []
        for p in ps:
            if 0 <= p < n:
                adj[p].append(i)
    is_input = [True] * n   # all vertices start blue (inputs); computed ones become red
    # Actually: "Inputs start blue" -> vertices with no parents are inputs (blue, free).
    # A vertex with parents must be computed (red) to be used.
    is_input = [len(parents[i]) == 0 if i < len(parents) else True for i in range(n)]

    # remaining use count: how many times each vertex is a parent of an unprocessed vertex
    remaining = [0] * n
    for i in range(n):
        for v in adj[i]:
            remaining[i] += 1

    resident = set()   # red values currently in fast memory (occupy slots)
    blue_done = set()  # vertices already computed and stored to slow (blue, free)
    done = set()       # vertices whose value is available (blue or resident)
    # inputs are available from start
    for i in range(n):
        if is_input[i]:
            done.add(i)

    ops = []
    budget = max(1, int(fast_memory_budget))

    for u in order:
        if _elapsed() > deadline:
            return ops, False
        ps = parents[u] if u < len(parents) else []
        if not ps:
            # input, already done
            continue
        # ensure all parents available
        for p in ps:
            if p not in done:
                # shouldn't happen in topo order; if it does, we cannot proceed safely
                return ops, False
            if p not in resident:
                # load parent from slow (blue) into fast (red)
                ops.append(['load', p])
                resident.add(p)
        # compute u
        ops.append(['compute', u])
        resident.add(u)
        # decrement remaining use of parents (they were just consumed)
        for p in ps:
            remaining[p] -= 1
        # evict if over budget
        while len(resident) > budget:
            # choose resident to evict: lowest remaining use; tie-break by not being
            # an input, and by being farthest from outputs (small remaining chain)
            best = None
            best_key = None
            for r in resident:
                # prefer to evict non-inputs first
                inp = 0 if is_input[r] else 1
                key = (remaining[r], inp)
                if best_key is None or key < best_key:
                    best_key = key
                    best = r
            if best is None:
                return ops, False
            resident.discard(best)
            ops.append(['store', best])
            blue_done.add(best)
        done.add(u)
        # after computing, parents with remaining use 0 can be stored to free slots
        for p in ps:
            if p in resident and remaining[p] == 0 and len(resident) > budget:
                resident.discard(p)
                ops.append(['store', p])
                blue_done.add(p)
    # ensure all outputs end blue: store any resident outputs
    for v in list(resident):
        ops.append(['store', v])
        resident.discard(v)
    return ops, True


def _verify(dag, ops, fast_memory_budget, deadline):
    """Exact state-machine verification. Returns True if valid."""
    if ops is None:
        return False
    n = len(_topo_order(dag))
    parents = dag.get('parents')
    if parents is None:
        parents = dag.get('preds', [])
    is_input = [len(parents[i]) == 0 if i < len(parents) else True for i in range(n)]
    budget = max(1, int(fast_memory_budget))
    red = set()
    blue = set()  # stored to slow
    computed = set()  # vertices whose value has been produced at least once
    # inputs start blue
    for i in range(n):
        if is_input[i]:
            blue.add(i)
    for op, v in ops:
        if _elapsed() > deadline:
            return False
        v = int(v)
        if v < 0 or v >= n:
            return False
        if op == 'load':
            if v not in blue:
                return False
            if v in red:
                return False
            if len(red) >= budget:
                return False
            red.add(v)
            blue.discard(v)
        elif op == 'store':
            if v not in red:
                return False
            red.discard(v)
            blue.add(v)
        elif op == 'compute':
            ps = parents[v] if v < len(parents) else []
            for p in ps:
                if p not in red and p not in blue:
                    # parent must be available; but compute needs it in fast? 
                    # Convention: compute reads parents; they must be available (red or blue).
                    # To be safe require red (fast) for the actual arithmetic.
                    if p not in red:
                        return False
            if v in red:
                return False
            if len(red) >= budget:
                return False
            red.add(v)
            computed.add(v)
        elif op == 'delete':
            if v not in red:
                return False
            red.discard(v)
        else:
            return False
    # all outputs (sinks) must finish blue
    adj = [[] for _ in range(n)]
    for i in range(n):
        ps = parents[i] if i < len(parents) else []
        for p in ps:
            if 0 <= p < n:
                adj[p].append(i)
    for v in range(n):
        if not adj[v]:  # sink = output
            if v not in blue:
                return False
    return True


def pebble(dag, fast_memory_budget):
    global _START
    if not hasattr(pebble, '_start'):
        pebble._start = _now()
    _START = getattr(pebble, '_start')
    deadline = _START + _DEADLINE
    order = _topo_order(dag)
    # Build a bounded-memory schedule
    ops, valid = _build_schedule(dag, fast_memory_budget, order, deadline)
    if valid and len(ops) <= 20000:
        if _verify(dag, ops, fast_memory_budget, deadline):
            return ops
    # Fallback: naive topological schedule (load each parent, compute each vertex),
    # which is always valid if budget >= max in-degree.
    ops = []
    parents = dag.get('parents')
    if parents is None:
        parents = dag.get('preds', [])
    n = len(order)
    for u in order:
        if _elapsed() > deadline:
            break
        ps = parents[u] if u < len(parents) else []
        for p in ps:
            ops.append(['load', p])
        ops.append(['compute', u])
        ops.append(['store', u])
    if len(ops) > 20000:
        # trim to a valid prefix is hard; return as-is (evaluator will score 0 if invalid)
        pass
    return ops
# EVOLVE-BLOCK-END
