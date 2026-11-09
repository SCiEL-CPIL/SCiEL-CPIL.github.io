"""Communication-minimizing DAG pebbling with deadline guard."""
import time

_START = time.time()
_DEADLINE = _START + 78.0  # 90s kill, keep 12s margin


def _remaining():
    return _DEADLINE - time.time()


def _topo_order(dag):
    """Return a topological order (parents before children)."""
    vertices = list(range(len(dag)))
    # dag[i] should list parents of i (topologically ordered parents)
    # Build children map
    n = len(dag)
    indeg = [0] * n
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in dag[v]:
            children[p].append(v)
            indeg[v] += 1
    from collections import deque
    q = deque([v for v in range(n) if indeg[v] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for w in children[u]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    if len(order) != n:
        # cycle or malformed; fallback
        order = list(range(n))
    return order


def _is_output(dag, v):
    """Output = no children (sink)."""
    n = len(dag)
    for w in range(n):
        for p in dag[w]:
            if p == v:
                return False
    return True


def _compute_naive(dag, budget):
    """Naive schedule: topological order, load each parent, compute, store each non-output result.
    Returns list of operations and total transfers."""
    order = _topo_order(dag)
    ops = []
    in_mem = set()
    for v in order:
        # load parents
        for p in dag[v]:
            if p not in in_mem:
                ops.append(["load", p])
                in_mem.add(p)
        # compute
        ops.append(["compute", v])
        # store non-output children (we store v if it has children and v is not output)
        # Actually we must keep v in memory if it's needed by children; naive keeps all.
        # For naive: store v if v has children (will be reloaded later) -> but naive keeps in memory.
        # Standard naive: load parents, compute, keep in memory (no store until end).
    # At end, outputs are already in memory (blue). No stores needed for outputs.
    # Naive transfers = loads only (parents loaded once) since we never store.
    return ops


def _simulate(dag, budget, ops):
    """Simulate operations, return (valid, transfers). Transfers = loads+stores."""
    n = len(dag)
    # state: 0=absent, 1=red(in fast mem), 2=blue(on disk)
    state = [2] * n  # inputs start blue (on disk)
    in_mem = 0
    transfers = 0
    for op, v in ops:
        if op == "load":
            if state[v] == 2:
                state[v] = 1
                in_mem += 1
                transfers += 1
            elif state[v] == 1:
                pass
            else:
                return False, 0
        elif op == "store":
            if state[v] == 1:
                state[v] = 2
                in_mem -= 1
                transfers += 1
            elif state[v] == 2:
                pass
            else:
                return False, 0
        elif op == "compute":
            # all parents must be in memory (red)
            for p in dag[v]:
                if state[p] != 1:
                    return False, 0
            # compute result: v becomes red in memory
            if state[v] == 1:
                # already in memory (recompute), no change
                pass
            else:
                state[v] = 1
                in_mem += 1
            if in_mem > budget:
                return False, 0
        elif op == "delete":
            if state[v] == 1:
                state[v] = 2
                in_mem -= 1
            elif state[v] == 2:
                pass
            else:
                return False, 0
        else:
            return False, 0
    # all outputs must be blue
    for v in range(n):
        if _is_output(dag, v) and state[v] != 2:
            return False, 0
    return True, transfers


def pebble(dag, fast_memory_budget):
    n = len(dag)
    if n == 0:
        return []

    order = _topo_order(dag)
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in dag[v]:
            children[p].append(v)

    # Identify inputs (no parents) and outputs (no children)
    inputs = [v for v in range(n) if len(dag[v]) == 0]
    outputs = [v for v in range(n) if len(children[v]) == 0]

    # Naive baseline: load each non-input vertex's parents as needed, compute in topo order,
    # keep everything in memory. Transfers = number of loads.
    # But we can do better by storing intermediate results to free memory.

    # Strategy: process in topological order. For each vertex, ensure parents are in memory,
    # compute, then if the vertex has children and memory is tight, store it.
    # Key insight: we only need to keep a vertex in memory while its children are being computed.
    # Once all children of v are computed, v can be stored (frees a slot).

    # Greedy: process in topo order, track which vertices are "active" (needed by future children).
    # A vertex v is active if any of its children haven't been computed yet.
    # We store v when it's no longer active.

    ops = []
    in_mem = set()
    computed = set()
    transfers = 0

    # Count how many children each vertex has remaining
    remaining_children = {v: len(children[v]) for v in range(n)}

    for v in order:
        if _remaining() < 2.0:
            break
        # Load parents not in memory
        for p in dag[v]:
            if p not in in_mem:
                ops.append(["load", p])
                in_mem.add(p)
                transfers += 1
        # Compute v
        ops.append(["compute", v])
        in_mem.add(v)
        computed.add(v)
        # Now decrement remaining_children for v's parents (they have one fewer child to wait for)
        for p in dag[v]:
            remaining_children[p] -= 1
            # If p is no longer needed (all children computed), store it to free memory
            if remaining_children[p] == 0 and p in in_mem and p != v:
                # But only if p is not an output (outputs must stay blue, not in memory)
                # Actually outputs should be stored at end. Let's store p if it has no more children.
                # But wait: p might still be needed if we haven't processed all its children.
                # remaining_children[p] == 0 means all children of p are computed, so p is no longer needed.
                ops.append(["store", p])
                in_mem.discard(p)
                transfers += 1
        # Check memory
        if len(in_mem) > fast_memory_budget:
            # Need to free some memory. Store any non-output, non-active vertex.
            # This shouldn't happen with the above strategy, but as a safeguard:
            # Find a vertex in memory that's no longer needed
            for u in list(in_mem):
                if u != v and remaining_children[u] == 0:
                    ops.append(["store", u])
                    in_mem.discard(u)
                    transfers += 1
                    if len(in_mem) <= fast_memory_budget:
                        break

    # At the end, all outputs must be blue (stored)
    for v in outputs:
        if v in in_mem:
            ops.append(["store", v])
            in_mem.discard(v)
            transfers += 1

    # Verify the schedule
    valid, actual_transfers = _simulate(dag, fast_memory_budget, ops)
    if not valid:
        # Fallback to simple naive: load all, compute all, store all
        ops = []
        in_mem = set()
        for v in order:
            for p in dag[v]:
                if p not in in_mem:
                    ops.append(["load", p])
                    in_mem.add(p)
            ops.append(["compute", v])
            in_mem.add(v)
        for v in range(n):
            if v in in_mem:
                ops.append(["store", v])
                in_mem.discard(v)
        valid, actual_transfers = _simulate(dag, fast_memory_budget, ops)
        if not valid:
            # Last resort: just return empty (will score 0 but avoid exception)
            return []

    return ops
