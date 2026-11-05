"""Rewritten program for the suite task; stronger composite priority heuristic."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 130.0  # safety margin well under the 150 s kill


def _compute_scores(times, preds, n):
    """Return a priority vector (higher = place earlier) for the instance."""
    times = np.asarray(times, dtype=float).reshape(-1)
    n = int(len(times))

    # Build adjacency: succ[i] = list of immediate successors of i.
    succ = [[] for _ in range(n)]
    pred = [[] for _ in range(n)]
    for i in range(n):
        for j in preds[i]:
            j = int(j)
            if 0 <= j < n:
                succ[j].append(i)
                pred[i].append(j)

    # Topological order (Kahn) and longest predecessor path (depth).
    indeg = [len(pred[i]) for i in range(n)]
    from collections import deque
    q = deque([i for i in range(n) if indeg[i] == 0])
    depth = np.zeros(n, dtype=float)  # longest path of predecessors ending at i
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succ[u]:
            # depth[v] = max(depth[v], depth[u] + times[u])
            cand = depth[u] + times[u]
            if cand > depth[v]:
                depth[v] = cand
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    # Downstream metrics: succ_work (sum of all successor times) and
    # succ_crit (longest path of successors). Compute in reverse topological order.
    succ_work = np.zeros(n, dtype=float)
    succ_crit = np.zeros(n, dtype=float)
    for u in reversed(order):
        w = 0.0
        c = 0.0
        for v in succ[u]:
            w += times[v] + succ_work[v]
            cv = times[v] + succ_crit[v]
            if cv > c:
                c = cv
        succ_work[u] = w
        succ_crit[u] = c

    out_deg = np.array([len(succ[i]) for i in range(n)], dtype=float)

    def _norm(a):
        a = np.asarray(a, dtype=float)
        m = float(a.max())
        if m <= 0.0:
            return np.zeros_like(a)
        return a / m

    # Weighted combination (scale-free).
    score = (
        1.0 * _norm(succ_work)
        + 0.8 * _norm(succ_crit)
        + 0.6 * _norm(out_deg)
        + 0.9 * _norm(times)
        + 0.2 * _norm(depth)
    )
    return score


def priority(instance):
    try:
        times = instance["times"]
        preds = instance["predecessors"]
        n = int(len(times))
        if n == 0:
            return np.asarray([], dtype=float)
        return _compute_scores(times, preds, n)
    except Exception:
        # Fallback to longest-task-first if anything unexpected happens.
        try:
            return np.asarray(instance["times"], dtype=float)
        except Exception:
            return None
# EVOLVE-BLOCK-END
