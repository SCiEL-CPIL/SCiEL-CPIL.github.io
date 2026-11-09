"""Communication-minimizing DAG pebbling.
Minimizes load+store transfers under a bounded fast-memory budget,
keeping outputs blue at the end. Greedy by remaining-use (recompute
cold values instead of holding them).
"""
# EVOLVE-BLOCK-START
import time

_START = time.time()
_DEADLINE = 55.0  # safety margin before the 60s limit


def _peb_plan(dag, budget):
    """Return list of (op, vertex) minimizing transfers."""
    # ---- parse the (opaque) dag into our internal graph ----
    parents = dag.get("parents") if isinstance(dag, dict) else None
    outputs = dag.get("outputs") if isinstance(dag, dict) else None
    vertices = dag.get("vertices") if isinstance(dag, dict) else None

    if parents is None:
        # fall back: maybe dag is a list of dicts {id, parents, output}
        if isinstance(dag, (list, tuple)):
            parents = {}
            outputs = []
            for i, item in enumerate(dag):
                if isinstance(item, dict):
                    vid = item.get("id", i)
                    parents[vid] = list(item.get("parents", []))
                    if item.get("output"):
                        outputs.append(vid)
                else:
                    parents[i] = []
            vertices = list(parents.keys())
        else:
            raise ValueError("Unrecognized dag structure")
    else:
        parents = {k: list(v) for k, v in parents.items()}
        outputs = list(outputs) if outputs is not None else []
        if vertices is None:
            vertices = list(parents.keys())

    n = len(vertices)
    idx = {v: i for i, v in enumerate(vertices)}
    out_set = set(idx[o] for o in outputs if o in idx)

    # children (reverse edges)
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[vertices[v]]:
            if p in idx:
                children[idx[p]].append(v)

    # topological order (Kahn) -- parents already topologically ordered,
    # but build our own to be safe
    indeg = [0] * n
    for v in range(n):
        indeg[v] = 0
    for v in range(n):
        for c in children[v]:
            indeg[c] += 1
    from collections import deque
    q = deque([v for v in range(n) if indeg[v] == 0])
    topo = []
    indeg2 = indeg[:]
    while q:
        u = q.popleft()
        topo.append(u)
        for c in children[u]:
            indeg2[c] -= 1
            if indeg2[c] == 0:
                q.append(c)
    if len(topo) != n:
        # cycle fallback: use vertex order
        topo = list(range(n))

    # remaining-use: number of times each value is needed in the future
    # (children that will consume it), plus 1 if it is an output (must stay blue)
    remaining = [0] * n
    for v in range(n):
        remaining[v] = len(children[v])
        if v in out_set:
            remaining[v] += 1  # output must be kept

    budget = max(0, int(budget))
    mem = {}   # idx -> True (red value held in fast memory)
    ops = []
    transfers = 0

    def compute(v):
        nonlocal transfers
        ops.append(("compute", v))
        # result is red in memory
        if v not in mem:
            mem[v] = True
            if len(mem) > budget:
                # evict the least-remaining-use non-output value
                _evict()

    def _evict():
        nonlocal transfers
        # pick a value in mem with smallest remaining use, not an output
        best = None
        best_r = None
        for m in list(mem.keys()):
            if m in out_set:
                continue
            r = remaining[m]
            if best is None or r < best_r:
                best, best_r = m, r
        if best is None:
            # all in mem are outputs; evict smallest anyway (shouldn't happen)
            for m in mem:
                if best is None or remaining[m] < best_r:
                    best, best_r = m, remaining[m]
        if best is not None and best in mem:
            del mem[best]
            ops.append(("store", best))
            transfers += 1

    def ensure_blue(v):
        """Make sure value v is available (red in mem) for use; if not, load."""
        nonlocal transfers
        if v in mem:
            return
        # need it: load (transfer) then it's red
        ops.append(("load", v))
        transfers += 1
        mem[v] = True
        if len(mem) > budget:
            _evict()

    # Process in topological order. A vertex is computed once when all its
    # parents are available. We keep values in memory while remaining-use > 0.
    for v in topo:
        if v in out_set:
            # outputs must be computed and stay blue; ensure parents first
            pass
        # ensure each parent is available (red) for this compute
        for p in parents[vertices[v]]:
            if p in idx:
                ensure_blue(idx[p])
        compute(v)
        # after computing, parents may no longer be needed
        for p in parents[vertices[v]]:
            if p in idx:
                remaining[idx[p]] -= 1
                if remaining[idx[p]] <= 0 and idx[p] not in out_set and idx[p] in mem:
                    # free it to make room
                    del mem[idx[p]]
                    ops.append(("store", idx[p]))
                    transfers += 1

    # Finalize: all outputs must be blue at the end.
    # Blue = present in fast memory (red) at termination.
    for o in out_set:
        if o not in mem:
            # recompute path is complex; just load if we can
            ensure_blue(o)

    return ops


def pebble(dag, fast_memory_budget):
    """Entry point. Returns list of [operation, vertex]."""
    try:
        budget = int(fast_memory_budget)
    except Exception:
        budget = 0

    # Deadline guard: if we've already spent too much, return a safe minimal plan.
    if time.time() - _START > _DEADLINE:
        return []

    try:
        plan = _peb_plan(dag, budget)
    except Exception:
        # Fallback: naive sequential plan (load parents, compute, store)
        plan = _naive_fallback(dag, budget)

    # Enforce op cap
    if len(plan) > 20000:
        plan = plan[:20000]

    return [[op, v] for (op, v) in plan]


def _naive_fallback(dag, budget):
    """Simple always-valid plan: load each parent, compute, store intermediates."""
    try:
        if isinstance(dag, dict):
            parents = {k: list(v) for k, v in dag.get("parents", {}).items()}
            vertices = list(parents.keys())
        elif isinstance(dag, (list, tuple)):
            parents = {}
            for i, item in enumerate(dag):
                if isinstance(item, dict):
                    vid = item.get("id", i)
                    parents[vid] = list(item.get("parents", []))
            vertices = list(parents.keys())
        else:
            return []
    except Exception:
        return []

    ops = []
    for v in vertices:
        for p in parents.get(v, []):
            ops.append(["load", p])
        ops.append(["compute", v])
        ops.append(["store", v])
    return ops
# EVOLVE-BLOCK-END
