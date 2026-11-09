# EVOLVE-BLOCK-START
import time
from collections import deque

import numpy as np

_DEADLINE = 110.0  # safety margin below the 120 s wall-clock limit


def priority(instance):
    t0 = time.time()

    # Robustly parse task times (the only field we truly need for a fallback).
    try:
        times = np.asarray(instance["times"], dtype=float)
        n = int(len(times))
    except Exception:
        return np.zeros(0, dtype=float)

    # LPT fallback (with distinct offsets) used if anything below fails.
    def _lpt():
        return np.asarray(times, dtype=float) + np.arange(n, dtype=float) * 1e-9

    try:
        preds_raw = instance["predecessors"]

        # Parse predecessors -> list of lists of valid int indices.
        preds = []
        for i in range(n):
            p = preds_raw[i]
            if p is None:
                preds.append([])
            elif isinstance(p, (list, tuple, np.ndarray)):
                preds.append([int(x) for x in p])
            else:
                preds.append([])

        # Build successor map (skip self-loops / out-of-range).
        successors = [[] for _ in range(n)]
        for i in range(n):
            for p in preds[i]:
                if 0 <= p < n and p != i:
                    successors[p].append(i)

        # In-degrees from the (validated) predecessor lists.
        indeg = [0] * n
        for i in range(n):
            for p in preds[i]:
                if 0 <= p < n and p != i:
                    indeg[i] += 1

        # Topological order (Kahn). Cycle-safe: append leftovers at the end.
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

        # Forward rank: longest path (in time) from task to any sink, incl. self.
        forward_rank = np.zeros(n, dtype=float)
        for u in reversed(topo):
            fr = times[u]
            for v in successors[u]:
                cand = times[u] + forward_rank[v]
                if cand > fr:
                    fr = cand
            forward_rank[u] = fr

        # Backward rank: longest path (in time) from any source to task, incl. self.
        backward_rank = np.zeros(n, dtype=float)
        for u in topo:
            br = times[u]
            for p in preds[u]:
                if 0 <= p < n and p != u:
                    cand = backward_rank[p] + times[u]
                    if cand > br:
                        br = cand
            backward_rank[u] = br

        # Total (transitive) descendant count via integer bitmasks (n=30 << 64).
        desc_mask = [0] * n
        for u in reversed(topo):
            m = 0
            for v in successors[u]:
                m |= (1 << v) | desc_mask[v]
            desc_mask[u] = m
        desc_count = np.array([bin(m).count("1") for m in desc_mask], dtype=float)

        # Combined priority:
        # - forward_rank * (desc_count + 1): emphasizes critical path length weighted by how many tasks depend on this one
        # - backward_rank: adds urgency from upstream (how much time has accumulated before this task)
        # - times: ensures longer tasks are prioritized (LPT effect)
        # This blends the current program's descendant-weighted critical path with
        # the backward-rank refinement from inspiration 1.
        base = forward_rank * (desc_count + 1.0) + backward_rank + times
        prio = base + np.arange(n, dtype=float) * 1e-9  # unique => no tie ambiguity

        if time.time() - t0 > _DEADLINE:
            return _lpt()
        return prio

    except Exception:
        return _lpt()
# EVOLVE-BLOCK-END
