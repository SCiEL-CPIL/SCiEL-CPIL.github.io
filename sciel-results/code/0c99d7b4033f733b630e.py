"""Assembly Line Balancing priority: bottom-rank (critical-path) heuristic."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    t0 = time.time()
    DEADLINE = 120.0 - 5.0  # safety margin before 120 s wall-clock kill

    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    ct = float(instance.get("cycle_time", 40))

    # Build successor adjacency from predecessors list.
    # predecessors[i] = list of tasks that must precede i
    preds = instance.get("predecessors", None)
    if preds is None:
        preds = [[] for _ in range(n)]

    succ = [[] for _ in range(n)]
    indeg = np.zeros(n, dtype=int)
    for i in range(n):
        plist = preds[i]
        for p in plist:
            succ[p].append(i)
            indeg[i] += 1

    # ---- Bottom rank: longest path from task i to any sink, inclusive of time[i] ----
    # Use memoized DFS with cycle guard.
    bottom = np.full(n, -1.0)
    visiting = np.zeros(n, dtype=bool)

    import sys
    sys.setrecursionlimit(max(10000, n * 4 + 100))

    def _dfs(i):
        if bottom[i] >= 0:
            return bottom[i]
        if time.time() > DEADLINE:
            return times[i]  # fallback: just own time
        visiting[i] = True
        best = 0.0
        for s in succ[i]:
            if not visiting[s]:
                best = max(best, _dfs(s))
        visiting[i] = False
        val = times[i] + best
        bottom[i] = val
        return val

    for i in range(n):
        _dfs(i)

    # ---- Auxiliary: number of immediate successors (more successors = earlier) ----
    succ_count = np.array([len(succ[i]) for i in range(n)], dtype=float)

    # ---- Auxiliary: topological position (earlier = smaller position) ----
    # Kahn topological order; position index.
    pos = np.full(n, n, dtype=float)
    indeg2 = indeg.copy()
    from collections import deque
    q = deque([i for i in range(n) if indeg2[i] == 0])
    order = 0
    while q:
        if time.time() > DEADLINE:
            break
        u = q.popleft()
        pos[u] = order
        order += 1
        for v in succ[u]:
            indeg2[v] -= 1
            if indeg2[v] == 0:
                q.append(v)
    # Normalize position to [0,1]
    pos_norm = pos / max(n - 1, 1)

    # ---- Combine: primary = bottom rank (higher first),
    #      tie-break: more successors, then earlier topological position.
    # Weights chosen so bottom rank dominates.
    # Scale succ_count and pos_norm to a small fraction of typical bottom rank.
    scale = max(float(np.max(bottom)) if n else 1.0, 1.0)
    w_succ = 0.05 * scale
    w_pos = 0.02 * scale

    score = bottom + w_succ * succ_count - w_pos * pos_norm

    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    return score
# EVOLVE-BLOCK-END
