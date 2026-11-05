"""Communication-minimizing DAG pebbling with bounded candidate interface."""
import time

# EVOLVE-BLOCK-START
import time as _time

# Hard wall-clock budget (seconds) with safety margin before the 90s kill.
_DEADLINE = 70.0
_start_time = None


def _now():
    return _time.time()


def _elapsed():
    global _start_time
    if _start_time is None:
        return 0.0
    return _time.time() - _start_time


def _time_left():
    return _DEADLINE - _elapsed()


def _build_graph(dag):
    """Parse dag into adjacency structures. Returns dict with keys:
       n, parents, children, inputs, outputs, topological order."""
    # dag is a dict-like mapping vertex -> list of parents (topologically ordered)
    # Determine vertex set and outputs.
    if hasattr(dag, "keys"):
        items = list(dag.items())
    else:
        # maybe list of (vertex, parents)
        items = list(dag)

    parents = {}
    for v, p in items:
        parents[v] = list(p)

    vertices = list(parents.keys())
    n = len(vertices)

    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p not in children:
                children[p] = []
            children[p].append(v)

    # Inputs: vertices with no parents (or explicitly marked).
    # Outputs: vertices with no children, OR explicitly marked in dag.
    # Some DAGs include an 'outputs' key; handle a few formats.
    inputs = [v for v in vertices if len(parents[v]) == 0]
    outputs = [v for v in vertices if len(children[v]) == 0]

    # If dag has explicit outputs marker, use it.
    if isinstance(dag, dict):
        if "outputs" in dag and not isinstance(dag["outputs"], (list, tuple)):
            pass
        # If a special key holds outputs as a list of vertices
        for key in ("outputs", "output", "sinks"):
            val = dag.get(key, None)
            if isinstance(val, (list, tuple)) and all(k in parents for k in val):
                outputs = list(val)
                break

    # Topological order (inputs first). Kahn's algorithm.
    indeg = {v: len(parents[v]) for v in vertices}
    from collections import deque
    q = deque([v for v in vertices if indeg[v] == 0])
    topo = []
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children[v]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    # Fallback if cycle or missing
    if len(topo) != n:
        topo = vertices

    return {
        "n": n,
        "parents": parents,
        "children": children,
        "inputs": inputs,
        "outputs": outputs,
        "topo": topo,
    }


def _naive_cost(g):
    """Naive schedule cost: load every input, store every output."""
    return len(g["inputs"]) + len(g["outputs"])


def _naive_schedule(g):
    """Naive: load all inputs, compute in topo order, store outputs.
       Returns list of (op, vertex)."""
    ops = []
    for v in g["inputs"]:
        ops.append(("load", v))
    for v in g["topo"]:
        if g["parents"][v]:
            ops.append(("compute", v))
    for v in g["outputs"]:
        ops.append(("store", v))
    return ops


def _simulate(ops, g, budget):
    """Simulate a schedule. Returns (ok, transfers, max_used).
       ok=False if invalid (compute with missing parent, over budget, etc.)."""
    fast = set()
    transfers = 0
    for op, v in ops:
        if op == "load":
            if v in fast:
                continue
            if len(fast) >= budget:
                return (False, transfers, len(fast))
            fast.add(v)
            transfers += 1
        elif op == "store":
            if v in fast:
                fast.discard(v)
                transfers += 1
        elif op == "delete":
            fast.discard(v)
        elif op == "compute":
            ps = g["parents"][v]
            for p in ps:
                if p not in fast:
                    return (False, transfers, len(fast))
            if len(fast) >= budget and v not in fast:
                return (False, transfers, len(fast))
            fast.add(v)
        else:
            return (False, transfers, len(fast))
    # All outputs must be blue at end -> present in fast.
    for v in g["outputs"]:
        if v not in fast:
            return (False, transfers, len(fast))
    return (True, transfers, len(fast))


def _pebble_dfs(g, budget):
    """Depth-first pebbling with recompute. Produces a valid schedule.
       Strategy: for each vertex in topo order, ensure its parents are in
       fast memory (loading as needed), compute it, then delete non-essential
       values to stay within budget. Keep outputs and values that will be
       reused."""
    parents = g["parents"]
    children = g["children"]
    topo = g["topo"]
    inputs = set(g["inputs"])
    outputs = set(g["outputs"])

    # Precompute for each vertex how many children remain (reuse count) to
    # decide whether to keep it in fast memory.
    ops = []
    fast = set()
    remaining_children = {v: len(children[v]) for v in topo}

    def ensure(v):
        """Ensure v is in fast; load if needed (inputs only)."""
        if v in fast:
            return
        if v in inputs:
            if len(fast) < budget:
                fast.add(v)
                ops.append(("load", v))
            else:
                # Need to free space: delete a non-output, non-needed value.
                # Find a deletable value (not output, not a parent of a not-yet-computed vertex that we still need).
                freed = False
                for cand in list(fast):
                    if cand in outputs:
                        continue
                    if cand in inputs:
                        # can delete and reload later
                        ops.append(("delete", cand))
                        fast.discard(cand)
                        freed = True
                        break
                if not freed:
                    # delete any non-output
                    for cand in list(fast):
                        if cand in outputs:
                            continue
                        ops.append(("delete", cand))
                        fast.discard(cand)
                        freed = True
                        break
            if len(fast) < budget:
                fast.add(v)
                ops.append(("load", v))

    for v in topo:
        ps = parents[v]
        if not ps:
            # input: load it (will be needed by children)
            ensure(v)
            continue
        # Ensure all parents are in fast
        for p in ps:
            ensure(p)
        # Compute v
        if len(fast) >= budget and v not in fast:
            # free space: delete a non-output value not needed soon
            for cand in list(fast):
                if cand in outputs:
                    continue
                # avoid deleting a parent we still need for this compute
                if cand in ps:
                    continue
                ops.append(("delete", cand))
                fast.discard(cand)
                break
        fast.add(v)
        ops.append(("compute", v))
        # After computing, decrement remaining children of parents; delete
        # parents that are no longer needed and not outputs.
        for p in ps:
            remaining_children[p] -= 1
            if remaining_children[p] == 0 and p not in outputs and p not in fast:
                pass
            if remaining_children[p] == 0 and p not in outputs:
                # Can delete p to free space (if we have room pressure).
                if len(fast) > budget:
                    ops.append(("delete", p))
                    fast.discard(p)
        # Keep v in fast if it has children remaining or is an output;
        # otherwise delete to free space.
        if remaining_children[v] == 0 and v not in outputs:
            # no future need; delete to free space
            ops.append(("delete", v))
            fast.discard(v)

    # Ensure all outputs are in fast at the end (load if somehow missing)
    for v in g["outputs"]:
        if v not in fast:
            # reload: it was computed before; we must have a way. As a safety,
            # load if input else we cannot recover -> but our schedule should
            # have kept outputs. If missing, attempt load (only valid for inputs).
            if v in inputs:
                ops.append(("load", v))
                fast.add(v)

    return ops


def _local_search(ops, g, budget, max_iters=200):
    """Try to reduce transfers via simple local moves. Returns best ops."""
    best_ops = ops
    best_cost, _, _ = _simulate(ops, g, budget)
    if not best_cost:
        best_cost = float("inf")

    for _ in range(max_iters):
        if _time_left() < 1.0:
            break
        improved = False
        n = len(best_ops)
        # Try deleting redundant loads (a load of v when v already in fast).
        for i in range(n):
            op, v = best_ops[i]
            if op != "load":
                continue
            cand = best_ops[:i] + best_ops[i+1:]
            ok, c, _ = _simulate(cand, g, budget)
            if ok and c < best_cost:
                best_ops = cand
                best_cost = c
                improved = True
                break
        if improved:
            continue
        # Try converting a store+later+load pattern: skip.
        break
    return best_ops


def pebble(dag, fast_memory_budget):
    global _start_time
    if _start_time is None:
        _start_time = _time.time()

    budget = int(fast_memory_budget)
    if budget < 1:
        budget = 1

    g = _build_graph(dag)
    naive_ops = _naive_schedule(g)
    naive_cost = _naive_cost(g)

    # Candidate 1: DFS recompute schedule.
    candidate_ops = _pebble_dfs(g, budget)
    ok, cost, _ = _simulate(candidate_ops, g, budget)
    if not ok:
        candidate_ops = naive_ops
        cost = naive_cost

    # Local search to trim redundant transfers.
    if _time_left() > 2.0:
        candidate_ops = _local_search(candidate_ops, g, budget)
        ok, cost, _ = _simulate(candidate_ops, g, budget)
        if not ok:
            candidate_ops = naive_ops
            cost = naive_cost

    # Fallback: if still worse than naive, use naive.
    if cost > naive_cost:
        candidate_ops = naive_ops

    # Enforce operation cap.
    if len(candidate_ops) > 20000:
        candidate_ops = candidate_ops[:20000]

    return candidate_ops
# EVOLVE-BLOCK-END
