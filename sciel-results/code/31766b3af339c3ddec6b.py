"""Rewritten program: position-aware priority for assembly line balancing."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _compute_priority(instance):
    times = np.asarray(instance["times"], float)
    n = int(times.size)

    preds = instance.get("predecessors")
    succs = instance.get("successors")

    # Build successor list from predecessors if only predecessors are given.
    if succs is None:
        if preds is None:
            preds = [[] for _ in range(n)]
        else:
            preds = [list(p) for p in preds]
        succs = [[] for _ in range(n)]
        for j in range(n):
            for i in preds[j]:
                if 0 <= i < n:
                    succs[i].append(j)
    else:
        if preds is None:
            preds = [[] for _ in range(n)]
        else:
            preds = [list(p) for p in preds]
        succs = [list(s) for s in succs]

    # Longest downstream (max path from task to a sink) and upstream (max path
    # from a source to task), via Kahn topological order.
    indeg = [len(preds[j]) for j in range(n)]
    queue = [j for j in range(n) if indeg[j] == 0]
    order = []
    while queue:
        j = queue.pop()
        order.append(j)
        for k in succs[j]:
            indeg[k] -= 1
            if indeg[k] == 0:
                queue.append(k)

    # Upstream longest path (including own time) in topological order.
    up = times.copy()
    for j in order:
        for i in preds[j]:
            if up[i] + times[j] > up[j]:
                up[j] = up[i] + times[j]

    # Downstream longest path (including own time) in reverse topological order.
    down = times.copy()
    for j in reversed(order):
        for k in succs[j]:
            if down[k] + times[j] > down[j]:
                down[j] = down[k] + times[j]

    total = float(times.sum())
    # Position in the project timeline: small = early (source-like), large = late.
    pos = up + down  # 2*up + (down - up) ~ position proxy; use up/down blend.
    pos = (up - 0.5 * times) + (down - 0.5 * times)  # span excluding self twice

    # Ranks in [0,1]: high = early/bottleneck (should go early).
    def rank_desc(arr):
        if n <= 1:
            return np.ones(n)
        order_idx = np.argsort(-arr, kind="stable")
        r = np.empty(n)
        r[order_idx] = np.arange(n)
        return r / (n - 1)

    # Early-position tasks (small pos) get high rank.
    rank_pos = 1.0 - rank_desc(pos)
    rank_time = rank_desc(times)
    # Upstream depth: deep tasks (many predecessors) should wait -> low priority.
    rank_up = 1.0 - rank_desc(up)

    # Blend: prioritize early bottlenecks and long tasks; penalize deep tasks.
    score = 1.0 * rank_pos + 0.6 * rank_time + 0.3 * rank_up
    # Add a small raw-time tiebreaker for stability.
    score = score + 1e-4 * (times / (total if total > 0 else 1.0))
    return score


def priority(instance):
    t0 = time.time()
    try:
        return _compute_priority(instance)
    except Exception:
        # Fallback to longest-task-first if analysis fails.
        return np.asarray(instance["times"], dtype=float)
# EVOLVE-BLOCK-END
