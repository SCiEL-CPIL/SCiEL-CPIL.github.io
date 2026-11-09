# EVOLVE-BLOCK-START
import time
from collections import deque

import numpy as np

_DEADLINE = 110.0  # safety margin below the 120 s wall-clock limit


def priority(instance):
    t0 = time.time()

    # Robustly parse task times
    try:
        times = np.asarray(instance["times"], dtype=float)
        n = int(len(times))
    except Exception:
        return np.zeros(0, dtype=float)

    # LPT fallback
    def _lpt():
        return np.asarray(times, dtype=float) + np.arange(n, dtype=float) * 1e-9

    try:
        preds_raw = instance["predecessors"]

        # Parse predecessors
        preds = []
        for i in range(n):
            p = preds_raw[i]
            if p is None:
                preds.append([])
            elif isinstance(p, (list, tuple, np.ndarray)):
                preds.append([int(x) for x in p])
            else:
                preds.append([])

        # Build successor map
        successors = [[] for _ in range(n)]
        for i in range(n):
            for p in preds[i]:
                if 0 <= p < n and p != i:
                    successors[p].append(i)

        # In-degrees
        indeg = [0] * n
        for i in range(n):
            for p in preds[i]:
                if 0 <= p < n and p != i:
                    indeg[i] += 1

        # Topological order (Kahn)
        queue = deque([i for i in range(n) if indeg[i] == 0])
        topo = []
        indeg2 = indeg[:]
        while queue:
            u = queue.popleft()
            topo.append(u)
            for v in successors[u]:
                indeg2[v] -= 1
                if indeg2[v] == 0:
                    queue.append(v)
        if len(topo) < n:
            seen = set(topo)
            for i in range(n):
                if i not in seen:
                    topo.append(i)

        # Forward rank: longest path from task to sink
        forward_rank = np.zeros(n, dtype=float)
        for u in reversed(topo):
            fr = times[u]
            for v in successors[u]:
                cand = times[u] + forward_rank[v]
                if cand > fr:
                    fr = cand
            forward_rank[u] = fr

        # Backward rank: longest path from source to task
        backward_rank = np.zeros(n, dtype=float)
        for u in topo:
            br = times[u]
            for p in preds[u]:
                if 0 <= p < n and p != u:
                    cand = backward_rank[p] + times[u]
                    if cand > br:
                        br = cand
            backward_rank[u] = br

        # Priority: Sum of forward and backward ranks.
        # This captures the total length of the critical path passing through the task.
        # It naturally emphasizes task duration (LPT) and criticality.
        prio = forward_rank + backward_rank
        prio += np.arange(n, dtype=float) * 1e-9  # unique tie-breaker

        if time.time() - t0 > _DEADLINE:
            return _lpt()
        return prio

    except Exception:
        return _lpt()
# EVOLVE-BLOCK-END
