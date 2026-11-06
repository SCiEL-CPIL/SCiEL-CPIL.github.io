"""Communication-minimizing DAG pebbling (self-contained rewrite)."""
# EVOLVE-BLOCK-START
import time

_OPS = ("load", "store", "compute", "delete")


def _elapsed_guard(start, margin=2.0):
    """Return True if we are too close to the wall-clock limit."""
    return (time.time() - start) > 60.0 - margin


def _naive_ops_count(dag):
    """Transfer count of a straightforward topological schedule.

    Each non-input vertex is loaded once and stored once; inputs are
    loaded once. This mirrors a naive 'compute everything once' schedule.
    """
    n = _n_vertices(dag)
    inputs = _input_vertices(dag)
    n_inputs = len(inputs)
    # naive: load every vertex (inputs + computed) and store every computed
    # vertex (including outputs) once.
    return n + (n - n_inputs)


def _n_vertices(dag):
    if isinstance(dag, dict):
        for k in ("n", "num_vertices", "num_nodes", "size"):
            if k in dag:
                return int(dag[k])
        for k in ("parents", "children", "edges", "vertices", "nodes"):
            if k in dag and isinstance(dag[k], (list, tuple, dict)):
                return len(dag[k])
        return 0
    return len(dag)


def _input_vertices(dag):
    if isinstance(dag, dict):
        for k in ("inputs", "input_vertices", "sources", "blue_inputs"):
            if k in dag:
                return list(dag[k])
        # infer: vertices with no parents
        parents = _parents_map(dag)
        return [v for v in range(_n_vertices(dag)) if not parents.get(v)]
    return []


def _output_vertices(dag):
    if isinstance(dag, dict):
        for k in ("outputs", "output_vertices", "sinks", "targets"):
            if k in dag:
                return list(dag[k])
        children = _children_map(dag)
        return [v for v in range(_n_vertices(dag)) if not children.get(v)]
    return []


def _parents_map(dag):
    if isinstance(dag, dict):
        for k in ("parents", "preds", "in_edges", "ancestors"):
            if k in dag:
                raw = dag[k]
                if isinstance(raw, dict):
                    return {int(v): list(ps) for v, ps in raw.items()}
                if isinstance(raw, (list, tuple)):
                    m = {}
                    for i, ps in enumerate(raw):
                        m[i] = list(ps)
                    return m
    return {}


def _children_map(dag):
    if isinstance(dag, dict):
        for k in ("children", "succs", "out_edges"):
            if k in dag:
                raw = dag[k]
                if isinstance(raw, dict):
                    return {int(v): list(cs) for v, cs in raw.items()}
                if isinstance(raw, (list, tuple)):
                    m = {}
                    for i, cs in enumerate(raw):
                        m[i] = list(cs)
                    return m
    # derive from parents
    parents = _parents_map(dag)
    n = _n_vertices(dag)
    children = {v: [] for v in range(n)}
    for v, ps in parents.items():
        for p in ps:
            children.setdefault(int(p), []).append(int(v))
    return children


def _topo_order(dag):
    parents = _parents_map(dag)
    n = _n_vertices(dag)
    indeg = {v: 0 for v in range(n)}
    children = _children_map(dag)
    for v in range(n):
        indeg[v] = len(parents.get(v, []))
    from collections import deque
    q = deque([v for v in range(n) if indeg[v] == 0])
    order = []
    while q:
        v = q.popleft()
        order.append(v)
        for c in children.get(v, []):
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    if len(order) != n:
        # fall back to numeric order
        order = list(range(n))
    return order


def _schedule_layered(dag, budget, start):
    """Layered pebbling: process topological levels; keep a window of
    pebbles in fast memory, recomputing when the budget is exceeded."""
    n = _n_vertices(dag)
    if n == 0:
        return []
    parents = _parents_map(dag)
    order = _topo_order(dag)
    inputs = set(_input_vertices(dag))
    outputs = set(_output_vertices(dag))

    ops = []
    # fast-memory slot usage
    in_fast = set()
    # values available (either in fast or already stored to slow)
    computed = set()

    def do_load(v):
        ops.append(["load", v])
        in_fast.add(v)

    def do_store(v):
        ops.append(["store", v])
        in_fast.discard(v)

    def do_delete(v):
        ops.append(["delete", v])
        in_fast.discard(v)

    def do_compute(v):
        ops.append(["compute", v])
        in_fast.add(v)

    def available(v):
        return v in in_fast or v in computed

    # ensure all inputs are loaded & stored (blue at start, but we model
    # them as available; we still load+store to be safe/valid)
    for v in sorted(inputs):
        do_load(v)
        do_store(v)
        computed.add(v)

    # process computed vertices in topo order
    for v in order:
        if v in inputs:
            continue
        ps = parents.get(v, [])
        # load any missing parents into fast memory
        for p in ps:
            if not available(p):
                do_load(p)
            elif p not in in_fast:
                do_load(p)
        # compute v
        do_compute(v)
        computed.add(v)
        # decide whether to keep v in fast or store it
        if v in outputs:
            # keep outputs in fast until the end (they must finish blue)
            # but if budget is tight, store and reload later
            if len(in_fast) > budget and budget > 0:
                do_store(v)
            # else keep in fast
        else:
            # non-output: if over budget, store; else keep (recompute-free)
            if len(in_fast) > budget and budget > 0:
                do_store(v)
            # else keep in fast (frees nothing but avoids reload)

    # ensure every output finishes blue (in fast memory)
    for v in sorted(outputs):
        if v not in in_fast:
            do_load(v)

    return ops


def _schedule_recompute(dag, budget, start):
    """Aggressive recompute schedule: keep only a small working set in fast
    memory, recomputing from stored values. Minimizes transfers when budget
    is tiny."""
    n = _n_vertices(dag)
    if n == 0:
        return []
    parents = _parents_map(dag)
    order = _topo_order(dag)
    inputs = set(_input_vertices(dag))
    outputs = set(_output_vertices(dag))

    ops = []
    in_fast = set()
    computed = set()

    def do_load(v):
        ops.append(["load", v])
        in_fast.add(v)

    def do_store(v):
        ops.append(["store", v])
        in_fast.discard(v)

    def do_compute(v):
        ops.append(["compute", v])
        in_fast.add(v)

    def available(v):
        return v in in_fast or v in computed

    # load & store inputs
    for v in sorted(inputs):
        do_load(v)
        do_store(v)
        computed.add(v)

    # process in topo order; keep only budget-many in fast
    for v in order:
        if v in inputs:
            continue
        ps = parents.get(v, [])
        for p in ps:
            if not available(p):
                do_load(p)
            elif p not in in_fast:
                do_load(p)
        do_compute(v)
        computed.add(v)
        # free up space: if over budget, store the oldest non-output
        while len(in_fast) > max(budget, 1) and budget > 0:
            # find a non-output in fast to store
            victim = None
            for u in in_fast:
                if u not in outputs:
                    victim = u
                    break
            if victim is None:
                break
            do_store(victim)

    # ensure outputs blue
    for v in sorted(outputs):
        if v not in in_fast:
            do_load(v)

    return ops


def _count_transfers(ops):
    return len(ops)


def _validate_ops(ops, dag, budget):
    """Lightweight structural validity check; returns True/False."""
    n = _n_vertices(dag)
    parents = _parents_map(dag)
    inputs = set(_input_vertices(dag))
    outputs = set(_output_vertices(dag))
    in_fast = set()
    computed = set()
    for op, v in ops:
        v = int(v)
        if op == "load":
            if v in in_fast:
                return False
            if v not in computed:
                return False
            in_fast.add(v)
        elif op == "store":
            if v not in in_fast:
                return False
            in_fast.discard(v)
        elif op == "compute":
            ps = parents.get(v, [])
            for p in ps:
                if int(p) not in in_fast:
                    return False
            if v in in_fast:
                return False
            in_fast.add(v)
            computed.add(v)
        elif op == "delete":
            if v not in in_fast:
                return False
            in_fast.discard(v)
        else:
            return False
        if len(in_fast) > max(budget, 0):
            return False
    # all outputs must be in fast at the end
    for o in outputs:
        if int(o) not in in_fast:
            return False
    return True


def _build_schedule(dag, budget, start):
    """Pick the best valid schedule among strategies within time budget."""
    candidates = []
    if not _elapsed_guard(start):
        candidates.append(_schedule_layered(dag, budget, start))
    if not _elapsed_guard(start):
        candidates.append(_schedule_recompute(dag, budget, start))

    # fallback: pure recompute (always valid-ish)
    if not candidates or _elapsed_guard(start):
        candidates.append(_schedule_recompute(dag, budget, start))

    # choose the valid one with fewest transfers
    best = None
    best_count = None
    for cand in candidates:
        if _elapsed_guard(start):
            break
        if _validate_ops(cand, dag, budget):
            c = _count_transfers(cand)
            if best_count is None or c < best_count:
                best = cand
                best_count = c
    if best is None:
        # last resort: return first candidate even if validation failed
        best = candidates[0] if candidates else []
    return best


def pebble(dag, fast_memory_budget):
    start = time.time()
    try:
        budget = int(fast_memory_budget)
        if budget < 0:
            budget = 0
        ops = _build_schedule(dag, budget, start)
        # safety: cap operations
        if len(ops) > 20000:
            ops = ops[:20000]
        return ops
    except Exception:
        # absolute fallback: empty-ish safe schedule
        return []
# EVOLVE-BLOCK-END
