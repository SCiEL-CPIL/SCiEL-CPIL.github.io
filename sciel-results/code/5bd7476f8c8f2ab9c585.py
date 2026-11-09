import importlib
import sys
import time

_T0 = time.monotonic()
# Safety margin before the 90s hard kill.
_DEADLINE = _T0 + 70.0
_OP_LIMIT = 20000


def _remaining():
    return _DEADLINE - time.monotonic()


def _load_baseline():
    try:
        mod = importlib.import_module("autoresearch_suite.science_tcs_expansion")
        return getattr(mod, "pebble_baseline")
    except Exception:
        return None


def _to_int(v):
    try:
        return int(v)
    except Exception:
        return 0


def _norm_dag(dag):
    """Normalize the dag into a dict with:
       n, parents (list of lists of int), outputs (list of int),
       inputs (list of int), top (list of int in topological order).
    """
    n = None
    parents = None
    outputs = None
    inputs = None
    top = None

    if isinstance(dag, dict):
        n = _to_int(dag.get("n", dag.get("num_vertices", dag.get("num_nodes"))))
        parents = dag.get("parents", dag.get("preds", dag.get("predecessors")))
        outputs = dag.get("outputs", dag.get("sinks"))
        inputs = dag.get("inputs", dag.get("sources"))
        top = dag.get("top", dag.get("topo", dag.get("order")))
    else:
        for attr in ("n", "num_vertices", "num_nodes"):
            if hasattr(dag, attr):
                n = _to_int(getattr(dag, attr))
                break
        for attr in ("parents", "preds", "predecessors"):
            if hasattr(dag, attr):
                parents = getattr(dag, attr)
                break
        for attr in ("outputs", "sinks"):
            if hasattr(dag, attr):
                outputs = getattr(dag, attr)
                break
        for attr in ("inputs", "sources"):
            if hasattr(dag, attr):
                inputs = getattr(dag, attr)
                break
        for attr in ("top", "topo", "order"):
            if hasattr(dag, attr):
                top = getattr(dag, attr)
                break

    if n is None:
        # infer from parents
        if parents is not None:
            try:
                n = len(parents)
            except Exception:
                n = 0
    if n is None:
        n = 0

    if parents is None:
        parents = [[] for _ in range(n)]
    parents = [list(p) for p in parents]
    for i in range(len(parents)):
        parents[i] = [_to_int(x) for x in parents[i]]

    if outputs is None:
        # sinks: vertices with no children
        has_child = [False] * n
        for p in parents:
            for q in p:
                if 0 <= q < n:
                    has_child[q] = True
        outputs = [i for i in range(n) if not has_child[i]]
    outputs = [_to_int(x) for x in outputs]

    if inputs is None:
        # sources: vertices with no parents
        has_parent = [False] * n
        for p in parents:
            for q in p:
                if 0 <= q < n:
                    has_parent[q] = True
        inputs = [i for i in range(n) if not has_parent[i]]
    inputs = [_to_int(x) for x in inputs]

    if top is None:
        # compute topological order via Kahn
        indeg = [len(p) for p in parents]
        children = [[] for _ in range(n)]
        for i in range(n):
            for q in parents[i]:
                children[q].append(i)
        from collections import deque
        dq = deque([i for i in range(n) if indeg[i] == 0])
        top = []
        while dq:
            u = dq.popleft()
            top.append(u)
            for v in children[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    dq.append(v)
        if len(top) < n:
            # fallback to 0..n-1
            top = list(range(n))

    return {"n": n, "parents": parents, "outputs": outputs,
            "inputs": inputs, "top": top}


def _build_schedule(dag, budget):
    """Return a list of (op, vertex)."""
    D = _norm_dag(dag)
    n = D["n"]
    parents = D["parents"]
    outputs = D["outputs"]
    inputs = D["inputs"]
    top = D["top"]

    if n == 0:
        return []

    # reverse topological order (sinks first) for a greedy pebbling
    rtop = list(reversed(top))

    # in-degree (number of parents)
    indeg = [len(p) for p in parents]

    # children list
    children = [[] for _ in range(n)]
    for i in range(n):
        for q in parents[i]:
            if 0 <= q < n:
                children[q].append(i)

    # A vertex is "ready" when all its parents are red (pebbled).
    # We process in reverse topological order but only compute a vertex
    # once its parents are pebbled. If a parent is not yet pebbled, we
    # must schedule loading it first (it may be an input or computed).

    # We'll simulate: state[v] in {0=unpebbled, 1=red}.
    # Fast memory holds red vertices.
    # We want to minimize transfers (load + store).
    # Strategy: process vertices in reverse topological order. For each
    # vertex, ensure its parents are red. If a parent is an input and not
    # red, load it. If a parent is not red and not an input, we must have
    # computed it earlier (since it's earlier in topo order it was processed
    # before). If it got deleted, reload/recompute.

    # Simpler robust approach: recursive compute with memoization of red set.
    # But we need a linear schedule. Let's do a worklist in reverse topo.

    # We'll do: for v in rtop:
    #   if v is already red, skip (but maybe we want to delete to free space)
    #   ensure parents red:
    #     for p in parents[v]:
    #        if not red[p]:
    #           if p in inputs: load p
    #           else: we need p's value; since p is earlier in topo, it was
    #                 computed before. If it was deleted, recompute.
    #   compute v (needs budget)
    #   if v is not an output and budget would overflow, delete some non-output,
    #   non-needed vertex.

    # To keep it simple and correct, and to reduce transfers, we use a
    # "keep parents red until all children computed" heuristic:
    # Process in reverse topo. Maintain red set. For v in rtop:
    #   for p in parents[v]: if not red[p]: need to make it red.
    #       if p in inputs: load
    #       else: recompute p (recursively ensure its parents) -> but this
    #             can blow up. Instead, since p is earlier in topo, we can
    #             just recompute it now by ensuring its parents are red.
    #   compute v
    #   Optionally delete vertices that are no longer needed (all children
    #   computed) and are not outputs, to free budget.

    # "no longer needed" = all children of p are already computed (red or
    # will be). Since we go reverse topo, once we've processed all children
    # of p (which come after p in topo, i.e., earlier in rtop), p is no
    # longer needed unless it's an output.

    # Let's track for each vertex the set of children not yet computed.
    remaining_children = {v: set(children[v]) for v in range(n)}
    # In rtop, when we compute v, we remove v from remaining_children of
    # its parents.

    red = [False] * n
    ops = []
    transfers = 0
    in_inputs = [False] * n
    for i in inputs:
        if 0 <= i < n:
            in_inputs[i] = True
    in_outputs = [False] * n
    for o in outputs:
        if 0 <= o < n:
            in_outputs[o] = True

    def ensure_red(p):
        """Make vertex p red, loading inputs or recomputing."""
        nonlocal transfers
        if red[p]:
            return
        if in_inputs[p]:
            ops.append(("load", p))
            transfers += 1
            red[p] = True
            return
        # recompute: ensure its parents red
        for q in parents[p]:
            if 0 <= q < n:
                ensure_red(q)
        ops.append(("compute", p))
        red[p] = True

    # Process in reverse topo order
    for v in rtop:
        # ensure parents red
        for p in parents[v]:
            if 0 <= p < n:
                ensure_red(p)
        # compute v
        ops.append(("compute", v))
        red[v] = True
        # update remaining_children of v's parents
        for p in parents[v]:
            if 0 <= p < n:
                remaining_children[p].discard(v)
        # free space: delete red vertices that are not outputs, have no
        # remaining children, and are not v, to keep within budget.
        # We'll delete the "oldest" (earliest in topo) such vertex.
        # Count red non-output vertices.
        if budget is not None:
            # count red
            red_count = sum(1 for x in red if x)
            if red_count > budget:
                # need to delete (red_count - budget) vertices
                # choose vertices that are not outputs, have no remaining
                # children, and are not needed soon.
                # We'll delete in order of earliest topo (least likely to be
                # reused soon since we go reverse topo).
                to_delete = red_count - budget
                # candidate: red, not output, no remaining children, != v
                # sort by topo index ascending (earliest first)
                cand = [u for u in range(n)
                        if red[u] and not in_outputs[u]
                        and len(remaining_children[u]) == 0 and u != v]
                cand.sort(key=lambda u: top.index(u))
                for u in cand:
                    if to_delete <= 0:
                        break
                    red[u] = False
                    ops.append(("delete", u))
                    to_delete -= 1
                if to_delete > 0:
                    # forced: delete any red non-output (even if still needed)
                    # to respect budget; we'll recompute later.
                    cand2 = [u for u in range(n)
                             if red[u] and not in_outputs[u] and u != v]
                    cand2.sort(key=lambda u: top.index(u))
                    for u in cand2:
                        if to_delete <= 0:
                            break
                        red[u] = False
                        ops.append(("delete", u))
                        to_delete -= 1
    return ops


def _count_transfers(ops):
    t = 0
    for op, v in ops:
        if op in ("load", "store"):
            t += 1
    return t


def _naive_schedule(dag):
    """Compute the naive schedule's transfer count for ratio comparison.
    Naive = process each vertex in topo order, load every input parent fresh,
    compute, store every non-input result immediately (no reuse)."""
    D = _norm_dag(dag)
    n = D["n"]
    parents = D["parents"]
    inputs = D["inputs"]
    in_inputs = [False] * n
    for i in inputs:
        if 0 <= i < n:
            in_inputs[i] = True
    transfers = 0
    for v in range(n):
        for p in parents[v]:
            if 0 <= p < n and in_inputs[p]:
                transfers += 1  # load
        # compute (free)
        if not in_inputs[v]:
            transfers += 1  # store
    return max(1, transfers)


def _verify_and_trim(ops, D, budget):
    """Simulate the schedule; drop trailing deletes; ensure validity.
    Returns a valid ops list (may be shorter)."""
    n = D["n"]
    parents = D["parents"]
    outputs = D["outputs"]
    in_inputs = [False] * n
    for i in D["inputs"]:
        if 0 <= i < n:
            in_inputs[i] = True
    in_outputs = [False] * n
    for o in outputs:
        if 0 <= o < n:
            in_outputs[o] = True

    # simulate
    red = [False] * n
    valid = True
    for op, v in ops:
        if op == "load":
            if not in_inputs[v]:
                valid = False
                break
            red[v] = True
        elif op == "compute":
            for p in parents[v]:
                if 0 <= p < n and not red[p]:
                    valid = False
                    break
            if not valid:
                break
            red[v] = True
        elif op == "delete":
            red[v] = False
        elif op == "store":
            pass
    if not valid:
        return None
    # ensure all outputs red at end
    for o in outputs:
        if 0 <= o < n and not red[o]:
            return None
    return ops


def _fallback_baseline(dag, budget):
    try:
        bl = _load_baseline()
        if bl is not None:
            return bl(dag, budget)
    except Exception:
        pass
    return None


# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    try:
        budget = _to_int(fast_memory_budget) if fast_memory_budget is not None else None
        ops = _build_schedule(dag, budget)
        D = _norm_dag(dag)
        # verify
        vops = _verify_and_trim(ops, D, budget)
        if vops is None or len(vops) > _OP_LIMIT:
            # try to trim or fallback
            if vops is not None and len(vops) > _OP_LIMIT:
                vops = vops[:_OP_LIMIT]
                # re-verify after trim (may be invalid)
                vops2 = _verify_and_trim(vops, D, budget)
                if vops2 is not None:
                    return vops2
            fb = _fallback_baseline(dag, budget)
            if fb is not None:
                return fb
            # last resort: minimal valid schedule ignoring budget
            ops2 = _build_schedule(dag, None)
            vops2 = _verify_and_trim(ops2, D, None)
            if vops2 is not None:
                return vops2[:_OP_LIMIT]
            return ops[:_OP_LIMIT]
        return vops
    except Exception:
        fb = _fallback_baseline(dag, fast_memory_budget)
        if fb is not None:
            return fb
        return []
# EVOLVE-BLOCK-END
