"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _topo_order(n, preds):
    """Kahn's algorithm; returns list of tasks in topological order."""
    indeg = np.zeros(n, dtype=int)
    succs = [[] for _ in range(n)]
    for i, ps in enumerate(preds):
        for p in ps:
            succs[p].append(i)
            indeg[i] += 1
    from collections import deque
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    # If cycle, append remaining
    if len(order) < n:
        remaining = [i for i in range(n) if i not in set(order)]
        order.extend(remaining)
    return order, succs


def priority(instance):
    t0 = time.time()
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    ct = float(instance["cycle_time"])

    # Build successor list and topological order
    order, succs = _topo_order(n, preds)

    # Longest path from each task to any sink (critical path remaining)
    # cp[i] = times[i] + max(cp[s] for s in succs[i], default=0)
    cp = np.zeros(n, dtype=float)
    # Process in reverse topological order
    for i in reversed(order):
        best = 0.0
        for s in succs[i]:
            if cp[s] > best:
                best = cp[s]
        cp[i] = times[i] + best

    # Number of successors (direct + transitive)
    trans_succ_count = np.zeros(n, dtype=int)
    for i in reversed(order):
        c = 0
        for s in succs[i]:
            c += 1 + trans_succ_count[s]
        trans_succ_count[i] = c

    # Longest path from source to each task (position in chain)
    lp = np.zeros(n, dtype=float)
    for i in order:
        best = 0.0
        for p in preds[i]:
            if lp[p] > best:
                best = lp[p]
        lp[i] = times[i] + best

    # Normalize components to [0,1] for blending
    def norm(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    cp_n = norm(cp)
    tsucc_n = norm(trans_succ_count.astype(float))
    lp_n = norm(lp)
    time_n = norm(times)

    # Slack-inspired term: tasks whose time is close to cycle_time are "tight"
    # We want to prioritize tasks that are large relative to cycle_time
    slack_term = times / ct  # larger = tighter fit

    # Combine: critical path dominates, then successor breadth, then time
    # Use weighted sum; critical path is most important for reducing stations
    score = (
        0.45 * cp_n
        + 0.20 * tsucc_n
        + 0.20 * time_n
        + 0.10 * lp_n
        + 0.05 * (slack_term / max(slack_term.max(), 1e-12))
    )

    # Safety: ensure we finish well before deadline
    if time.time() - t0 > 100:
        # Fallback to simple time-based priority
        return np.asarray(times, dtype=float)

    return np.asarray(score, dtype=float)
# EVOLVE-BLOCK-END
