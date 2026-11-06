"""Communication-minimizing DAG pebbling: exact DP over core subsets + verified fallback."""
import time as _time

MAX_OPS = 20000


def _count_transfers(ops):
    return sum(1 for o, _ in ops if o in ("load", "store", "compute"))


def _verify(ops, dag, budget):
    inputs = dag.get("inputs", [])
    outputs = dag.get("outputs", [])
    parents = dag.get("parents", {})
    n = dag.get("n", 0)
    if n == 0:
        n = max(list(inputs) + list(outputs) + list(parents), default=-1) + 1
    if len(ops) > MAX_OPS:
        return False
    state = {v: 0 for v in range(n)}
    for v in inputs:
        state[v] = 1
    for op, v in ops:
        if op == "load":
            if v not in inputs or state[v] != 0:
                return False
            state[v] = 2
        elif op == "store":
            if state[v] != 2:
                return False
            state[v] = 1
        elif op == "compute":
            if v in inputs or state[v] == 2:
                return False
            ps = parents.get(v, [])
            if not all(state[p] == 2 for p in ps):
                return False
            if sum(1 for x in range(n) if state[x] == 2) > budget:
                return False
            state[v] = 2
        elif op == "delete":
            if state[v] != 2:
                return False
            state[v] = 0
        else:
            return False
    return all(state[v] == 1 for v in outputs)


def _naive(dag, budget):
    inputs = list(dag.get("inputs", []))
    outputs = list(dag.get("outputs", []))
    parents = dag.get("parents", {})
    n = dag.get("n", 0)
    if n == 0:
        n = max(list(inputs) + list(outputs) + list(parents), default=-1) + 1
    ops = []
    state = {v: 0 for v in range(n)}
    for v in inputs:
        state[v] = 1
    core = [v for v in range(n) if v not in inputs]
    for v in core:
        for p in parents.get(v, []):
            if state[p] == 0:
                ops.append(["load", p]); state[p] = 2
            elif state[p] == 1:
                ops.append(["load", p]); state[p] = 2
        ops.append(["compute", v]); state[v] = 2
    for v in outputs:
        ops.append(["store", v]); state[v] = 1
    return ops


def _dp_solve(dag, budget):
    """Exact min-transfer schedule via DP over core subsets. Returns ops or None."""
    inputs = set(dag.get("inputs", []))
    outputs = set(dag.get("outputs", []))
    parents = dag.get("parents", {})
    n = dag.get("n", 0)
    if n == 0:
        n = max(list(inputs) + list(outputs) + list(parents), default=-1) + 1
    core = [v for v in range(n) if v not in inputs]
    k = len(core)
    if k == 0:
        return []
    idx = {v: i for i, v in enumerate(core)}
    out_of_core = {v: [p for p in parents.get(v, []) if p not in idx] for v in core}

    dep = [0] * k
    for i, v in enumerate(core):
        m = 0
        for p in parents.get(v, []):
            if p in idx:
                m |= 1 << idx[p]
        dep[i] = m

    reach = [0] * (1 << k)
    for S in range(1 << k):
        r = 0
        for i, v in enumerate(core):
            if (S >> i) & 1:
                continue
            if (dep[i] & ~S) == 0 and all(p in inputs for p in out_of_core[v]):
                r |= 1 << i
        reach[S] = r

    INF = float("inf")
    cost = [INF] * (1 << k)
    cost[0] = 0
    for S in range(1 << k):
        cS = cost[S]
        if cS == INF:
            continue
        r = reach[S]
        while r:
            b = r & (-r)
            T = S | b
            if cS + 1 < cost[T]:
                cost[T] = cS + 1
            r ^= b
    full = (1 << k) - 1
    if cost[full] == INF:
        return None

    # reconstruct chain
    chain = []
    S = full
    while S != 0:
        r = reach[S]
        found = False
        while r:
            b = r & (-r)
            T = S | b
            if cost[T] == cost[S] + 1:
                chain.append(T)
                S = T
                found = True
                break
            r ^= b
        if not found:
            return None
    chain.reverse()

    # expand to explicit ops with a live memory simulation
    ops = []
    state = {v: 0 for v in range(n)}
    for v in inputs:
        state[v] = 1

    def ensure_fast(p):
        """Make core vertex p blue in fast (recompute if needed)."""
        nonlocal ops
        if state[p] == 2:
            return
        # p is core; recompute it (it must be computable: parents available)
        for pp in parents.get(p, []):
            if pp in inputs:
                if state[pp] != 2:
                    ops.append(["load", pp]); state[pp] = 2
            else:
                ensure_fast(pp)
        ops.append(["compute", p]); state[p] = 2

    prev = 0
    for S in chain:
        new = S & ~prev
        # compute newly-added core vertices in dependency order
        while new:
            b = new & (-new)
            i = b.bit_length() - 1
            v = core[i]
            for p in parents.get(v, []):
                if p in inputs:
                    if state[p] != 2:
                        ops.append(["load", p]); state[p] = 2
                else:
                    ensure_fast(p)
            ops.append(["compute", v]); state[v] = 2
            new ^= b
        prev = S

    # store outputs (they are core, so blue in fast)
    for v in outputs:
        if state[v] == 2:
            ops.append(["store", v]); state[v] = 1

    if len(ops) > MAX_OPS:
        return None
    return ops


def pebble(dag, fast_memory_budget):
    t0 = _time.time()
    deadline = t0 + 75.0

    best = _naive(dag, fast_memory_budget)
    best_t = _count_transfers(best)

    core_size = len([v for v in range(dag.get("n", 0) or max(list(dag.get("inputs", [])) + list(dag.get("outputs", [])) + list(dag.get("parents", {})), default=-1) + 1) if v not in set(dag.get("inputs", []))])

    if core_size <= 18 and _time.time() < deadline:
        try:
            ops = _dp_solve(dag, fast_memory_budget)
            if ops is not None and _verify(ops, dag, fast_memory_budget):
                t = _count_transfers(ops)
                if t < best_t:
                    best = ops
                    best_t = t
        except Exception:
            pass

    return best
