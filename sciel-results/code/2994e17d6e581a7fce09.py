"""Assembly Line Balancing: rank tasks by longest path to end, tie-break by successors then time."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _build_graph(instance):
    n = len(instance["times"])
    preds = instance["predecessors"]
    # successors[j] = list of tasks that have j as a predecessor
    succ = [[] for _ in range(n)]
    indeg = [0] * n
    for j in range(n):
        ps = preds[j] if preds is not None else []
        if ps is None:
            ps = []
        for p in ps:
            succ[p].append(j)
            indeg[j] += 1
    return succ, indeg


def _longest_path_to_end(succ, times, n):
    # lp[j] = times[j] + max(lp[s] for s in succ[j])
    # Compute via reverse topological order (process from sinks backward)
    # Use Kahn's algorithm on the reversed graph to get an order, then reverse.
    # Simpler: iterative DP with memoization via topological order.
    # Compute topological order of original graph.
    indeg = [0] * n
    for j in range(n):
        for s in succ[j]:
            indeg[s] += 1
    from collections import deque
    q = deque([j for j in range(n) if indeg[j] == 0])
    topo = []
    while q:
        j = q.popleft()
        topo.append(j)
        for s in succ[j]:
            indeg[s] -= 1
            if indeg[s] == 0:
                q.append(s)
    # If cycle, topo may be incomplete; handle gracefully.
    lp = np.zeros(n, dtype=float)
    # Process in reverse topological order
    for j in reversed(topo):
        best = 0.0
        for s in succ[j]:
            if lp[s] > best:
                best = lp[s]
        lp[j] = times[j] + best
    # Tasks not in topo (cycle members) get just their own time
    for j in range(n):
        if lp[j] == 0.0:
            lp[j] = times[j]
    return lp


def priority(instance):
    start = time.time()
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    preds = instance["predecessors"]
    succ, _ = _build_graph(instance)
    lp = _longest_path_to_end(succ, times, n)
    # Number of successors (immediate) - more successors = earlier in chain = higher priority
    nsucc = np.array([len(s) for s in succ], dtype=float)
    # Primary: longest path to end (descending). Secondary: more successors. Tertiary: longer time.
    # Combine into a single score so that higher = higher priority.
    # Normalize components to [0,1] to balance magnitudes.
    def norm(x):
        x = np.asarray(x, dtype=float)
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    s_lp = norm(lp)
    s_ns = norm(nsucc)
    s_t = norm(times)
    score = 0.6 * s_lp + 0.25 * s_ns + 0.15 * s_t
    # Safety margin guard
    if time.time() - start > 100:
        pass
    return score
# EVOLVE-BLOCK-END
