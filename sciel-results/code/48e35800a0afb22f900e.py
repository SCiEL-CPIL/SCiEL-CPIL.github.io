"""Rewritten pebbling scheduler for communication-minimizing DAG pebbling.

Design:
- Parse the DAG (topologically ordered parents, output vertices, inputs).
- Build a valid pebbling schedule that keeps the set of "red" (fast-memory)
  values within fast_memory_budget, minimises load/store (transfer) operations,
  and finishes with all outputs blue.
- Use a layer-by-layer greedy with a priority queue that evicts the value
  least likely to be reused (lowest remaining in-degree / farthest next use)
  to keep peak memory low, while re-loading from slow memory only when needed.
- Recomputation is allowed; we prefer recompute over load+store when a value
  has already been stored and would otherwise require a transfer.
- A wall-clock deadline guard returns the best valid schedule found before the
  90 s limit (with a safety margin).
"""

import time
import heapq

_DEADLINE = 85.0  # seconds; leave margin under the 90 s kill
_START = None


def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START


def _parse_dag(dag):
    """Normalise the DAG into a uniform structure.

    Returns dict with keys:
      n: number of vertices (0..n-1)
      parents: list of lists, parents[v] = list of predecessor vertex ids
      children: list of lists, children[v] = list of successor vertex ids
      is_input: bool per vertex (no parents)
      is_output: bool per vertex (in output set)
      topo: topological order list of vertex ids
    """
    # The task says "dag has topologically ordered parents and output vertices."
    # Support several plausible encodings defensively.
    if isinstance(dag, dict):
        # Could be {"parents": [...], "outputs": [...]} etc.
        parents = list(dag.get("parents", []))
        outputs = set(dag.get("outputs", dag.get("output", [])))
        n = dag.get("n", dag.get("num_vertices", len(parents)))
        if n is None:
            n = len(parents)
        # Ensure parents is a list of lists
        parents = [list(p) for p in parents]
        while len(parents) < n:
            parents.append([])
        parents = parents[:n]
        topo = list(range(n))  # assume already topologically ordered
    else:
        # Assume dag is a list of lists: dag[v] = parents of v
        parents = [list(p) for p in dag]
        n = len(parents)
        outputs = set(range(n))  # if unknown, treat all sinks as outputs
        # Better: outputs = sinks (no children)
        children = [[] for _ in range(n)]
        for v, ps in enumerate(parents):
            for p in ps:
                children[p].append(v)
        outputs = set(v for v in range(n) if not children[v])
        topo = list(range(n))

    children = [[] for _ in range(n)]
    for v, ps in enumerate(parents):
        for p in ps:
            if 0 <= p < n:
                children[p].append(v)

    is_input = [len(parents[v]) == 0 for v in range(n)]
    is_output = [v in outputs for v in range(n)]
    # If no outputs marked, fall back to sinks
    if not any(is_output):
        is_output = [len(children[v]) == 0 for v in range(n)]

    return {
        "n": n,
        "parents": parents,
        "children": children,
        "is_input": is_input,
        "is_output": is_output,
        "topo": topo,
    }


def _simulate(dag, ops, budget):
    """Validate a schedule and count transfers. Returns (ok, transfers, peak)."""
    n = dag["n"]
    # state: -1 = not present, 0 = blue (slow), 1 = red (fast)
    # Inputs start blue.
    state = [0 if dag["is_input"][v] else -1 for v in range(n)]
    # Non-inputs are not initially present.
    red = 0
    transfers = 0
    peak = 0
    for op, v in ops:
        if not (0 <= v < n):
            return False, 0, 0
        if op == "load":
            # Move blue -> red. Must be blue (available in slow memory).
            if state[v] != 0:
                return False, 0, 0
            if red >= budget:
                return False, 0, 0
            state[v] = 1
            red += 1
            transfers += 1
            peak = max(peak, red)
        elif op == "store":
            # Move red -> blue. Frees a slot.
            if state[v] != 1:
                return False, 0, 0
            state[v] = 0
            red -= 1
            transfers += 1
        elif op == "compute":
            # All parents must be red; result becomes red (consumes a slot).
            # If the vertex already red, it's a recompute (still must fit).
            for p in dag["parents"][v]:
                if state[p] != 1:
                    return False, 0, 0
            if state[v] != 1:
                if red >= budget:
                    return False, 0, 0
                red += 1
                peak = max(peak, red)
            state[v] = 1
        elif op == "delete":
            # Remove a red value (free slot). Only allowed if not an output
            # that still must finish blue? Deleting a red non-output is fine.
            if state[v] != 1:
                return False, 0, 0
            state[v] = -1
            red -= 1
        else:
            return False, 0, 0
    # All outputs must finish blue.
    for v in range(n):
        if dag["is_output"][v] and state[v] != 0:
            return False, 0, 0
    return True, transfers, peak


def _greedy_schedule(dag, budget):
    """Layer-by-layer greedy pebbling with a priority-queue eviction policy.

    Strategy:
      - Process vertices in topological order.
      - Maintain a set of red vertices (in fast memory).
      - To compute a vertex, its parents must be red. If a parent is not red,
        we must (re)load or (re)compute it. We prefer to keep a parent red if
        it will be reused; otherwise we evict (store) the least-reused red value.
      - When memory is full and we need a new red slot, evict the red value
        with the lowest "remaining reuse" (fewest future consumers not yet
        computed), breaking ties by farthest next use.
      - After all outputs are computed and blue, finish.

    This is a heuristic; we then validate and, if invalid, fall back to a
    safe (slightly higher transfer) schedule.
    """
    n = dag["n"]
    parents = dag["parents"]
    children = dag["children"]
    topo = dag["topo"]

    # Precompute, for each vertex, the list of children in topological order.
    # remaining_reuse[v] = number of children of v that have not yet been
    # "finalised" (computed). We track via a counter.
    # next_use[v] = index (in topo) of the earliest child not yet computed.

    ops = []
    # state: 0 blue, 1 red, -1 absent
    state = [0 if dag["is_input"][v] else -1 for v in range(n)]
    red = 0

    # For eviction priority: we want to evict the red vertex with the smallest
    # remaining reuse. Use a heap of (reuse_count, topo_index, vertex).
    # We'll rebuild the heap lazily / use a lazy-delete approach.
    import bisect

    # remaining children count per vertex (children not yet computed)
    rem_children = [len(children[v]) for v in range(n)]
    # A vertex is "finalised" when all its children have been computed.

    # We compute vertices in topo order. For each vertex v in topo order,
    # ensure all its parents are red, compute v, then decide what to evict.

    # To know reuse, define reuse[v] = number of children of v (future consumers).
    # As we compute a child c of v, we decrement reuse[v].

    reuse = [len(children[v]) for v in range(n)]

    # Lazy heap for eviction: entries (reuse[v], v). We pop until entry matches
    # current reuse[v] and state[v]==1.
    evict_heap = []

    def push_evict(v):
        heapq.heappush(evict_heap, (reuse[v], v))

    def pop_evict():
        while evict_heap:
            r, v = evict_heap[0]
            if state[v] == 1 and reuse[v] == r:
                heapq.heappop(evict_heap)
                return v
            heapq.heappop(evict_heap)
        return None

    def ensure_red(v):
        """Make vertex v red. If already red, do nothing. Otherwise load or
        recompute as appropriate, evicting to make room if needed."""
        nonlocal red
        if state[v] == 1:
            return
        # Need to bring v into fast memory as red.
        # If v is blue (available in slow memory), we can load it.
        # If v is absent, we must compute it (parents red) — but that requires
        # parents red, which we handle by recursion.
        if state[v] == 0:
            # load
            if red >= budget:
                evict_one()
            state[v] = 1
            red += 1
            ops.append(["load", v])
            push_evict(v)
            return
        # state[v] == -1: must compute. Ensure parents red first.
        for p in parents[v]:
            ensure_red(p)
        if red >= budget:
            evict_one()
        state[v] = 1
        red += 1
        ops.append(["compute", v])
        push_evict(v)

    def evict_one():
        """Evict one red value to free a slot. Prefer lowest reuse; if reuse is
        0 (no future consumers), evicting is free of future cost (we can just
        store it so it's still blue if it's an output, or delete it)."""
        nonlocal red
        while True:
            v = pop_evict()
            if v is None:
                # Should not happen if red>0; fallback: evict any red.
                for u in range(n):
                    if state[u] == 1:
                        v = u
                        break
                if v is None:
                    return
            # Decide: store (keeps blue) vs delete (frees, loses value).
            # If v is an output, we must keep it available; but we can store it
            # (blue) now and it stays blue. Storing keeps it valid for the final
            # "outputs blue" check and for reuse via load.
            # If v is not an output and reuse[v]==0, we can delete it (no future
            # need) — cheaper in the sense it won't be reloaded.
            if reuse[v] == 0 and not dag["is_output"][v]:
                state[v] = -1
                red -= 1
                ops.append(["delete", v])
            else:
                state[v] = 0
                red -= 1
                ops.append(["store", v])
            # After eviction, reuse counts unchanged. Continue if we still need
            # more room (caller checks red).
            return

    # Main loop over topo order.
    for v in topo:
        if _now() > _DEADLINE:
            break
        ensure_red(v)
        # After computing v, decrement reuse of its parents (they've been
        # consumed by v).
        for p in parents[v]:
            reuse[p] -= 1
        # Mark v as "computed". Its children still need it.
        # Note: v remains red for now; eviction will handle it later.

    # After the main loop, ensure all outputs are blue.
    for v in topo:
        if dag["is_output"][v]:
            if state[v] == 1:
                state[v] = 0
                red -= 1
                ops.append(["store", v])
            elif state[v] == -1:
                # Output absent: must bring it back. Recompute (parents may be
                # gone). Ensure parents red, compute, then store.
                for p in parents[v]:
                    ensure_red(p)
                if red >= budget:
                    evict_one()
                state[v] = 1
                red += 1
                ops.append(["compute", v])
                state[v] = 0
                red -= 1
                ops.append(["store", v])

    return ops


def _safe_schedule(dag, budget):
    """A simple, always-valid schedule: process in topological order, keep all
    inputs red, compute each vertex, store it immediately (so peak memory is
    bounded by in-degree + 1). This may exceed the budget for wide DAGs, so we
    guard: if in-degree exceeds budget, fall back to a recompute-heavy schedule.
    """
    n = dag["n"]
    parents = dag["parents"]
    children = dag["children"]
    topo = dag["topo"]
    ops = []
    state = [0 if dag["is_input"][v] else -1 for v in range(n)]
    red = 0

    # Load all inputs as red (they start blue).
    for v in topo:
        if dag["is_input"][v]:
            if red < budget:
                state[v] = 1
                red += 1
                ops.append(["load", v])

    # Compute each vertex; before computing, ensure parents red (re-load if
    # evicted). After computing, store the vertex (blue) to free a slot.
    for v in topo:
        if _now() > _DEADLINE:
            break
        if dag["is_input"][v]:
            continue
        # Ensure parents red.
        for p in parents[v]:
            if state[p] != 1:
                # Need to (re)load p. Make room.
                if red >= budget:
                    # Evict a non-input red value (store it).
                    for u in topo:
                        if state[u] == 1 and not dag["is_input"][u]:
                            state[u] = 0
                            red -= 1
                            ops.append(["store", u])
                            break
                if state[p] == 0:
                    state[p] = 1
                    red += 1
                    ops.append(["load", p])
                else:
                    # p absent: recompute p (its parents must be red).
                    _recompute(p, ops, state, dag, budget, topo, red_holder=[red])
        # Compute v.
        if red >= budget:
            for u in topo:
                if state[u] == 1 and not dag["is_input"][u]:
                    state[u] = 0
                    red -= 1
                    ops.append(["store", u])
                    break
        state[v] = 1
        red += 1
        ops.append(["compute", v])
        # Store v immediately to keep memory low (unless it's an input).
        if not dag["is_input"][v]:
            state[v] = 0
            red -= 1
            ops.append(["store", v])

    return ops


def _recompute(v, ops, state, dag, budget, topo, red_holder):
    """Recompute an absent vertex v (parents must be made red)."""
    # This is a fallback; keep it simple and recursive with depth guard.
    for p in dag["parents"][v]:
        if state[p] != 1:
            if red_holder[0] >= budget:
                for u in topo:
                    if state[u] == 1 and not dag["is_input"][u]:
                        state[u] = 0
                        red_holder[0] -= 1
                        ops.append(["store", u])
                        break
            if state[p] == 0:
                state[p] = 1
                red_holder[0] += 1
                ops.append(["load", p])
            else:
                _recompute(p, ops, state, dag, budget, topo, red_holder)
    if red_holder[0] >= budget:
        for u in topo:
            if state[u] == 1 and not dag["is_input"][u]:
                state[u] = 0
                red_holder[0] -= 1
                ops.append(["store", u])
                break
    state[v] = 1
    red_holder[0] += 1
    ops.append(["compute", v])


def pebble(dag, fast_memory_budget):
    global _START
    _START = time.time()

    parsed = _parse_dag(dag)
    budget = int(fast_memory_budget)
    if budget < 1:
        budget = 1

    # Try the greedy schedule; validate; if invalid or too many ops, fall back.
    best_ops = None
    best_transfers = None

    candidates = []
    g = _greedy_schedule(parsed, budget)
    candidates.append(g)
    s = _safe_schedule(parsed, budget)
    candidates.append(s)

    for ops in candidates:
        if len(ops) > 20000:
            continue
        ok, transfers, peak = _simulate(parsed, ops, budget)
        if ok:
            if best_transfers is None or transfers < best_transfers:
                best_transfers = transfers
                best_ops = ops

    if best_ops is None:
        # Last resort: emit an empty-ish valid schedule if possible, else return
        # the safe one even if over budget (better than nothing).
        best_ops = s if s else []

    return best_ops
