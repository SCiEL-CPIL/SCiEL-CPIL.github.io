"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _topo_order(n, preds):
    """Return a topological ordering of tasks 0..n-1 given predecessor lists."""
    indeg = [0] * n
    succ = [[] for _ in range(n)]
    for j in range(n):
        for p in preds[j]:
            if 0 <= p < n:
                indeg[j] += 1
                succ[p].append(j)
    from collections import deque
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    # If not all nodes visited (cycle), append remaining
    if len(order) < n:
        seen = set(order)
        for i in range(n):
            if i not in seen:
                order.append(i)
    return order


def priority(instance):
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s

    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)

    # Normalize predecessor lists
    preds_list = []
    for j in range(n):
        p = preds[j]
        if isinstance(p, (list, tuple, np.ndarray)):
            preds_list.append([int(x) for x in p])
        else:
            preds_list.append([])

    # Build successor lists
    succ = [[] for _ in range(n)]
    for j in range(n):
        for p in preds_list[j]:
            if 0 <= p < n:
                succ[p].append(j)

    # Topological order
    order = _topo_order(n, preds_list)
    rev_order = list(reversed(order))

    # Forward rank: longest path from any start to j (including j's own time)
    fwd_rank = np.zeros(n, dtype=float)
    for u in order:
        best = 0.0
        for p in preds_list[u]:
            if fwd_rank[p] > best:
                best = fwd_rank[p]
        fwd_rank[u] = best + times[u]

    # Backward rank: longest path from j to any end (including j's own time)
    bwd_rank = np.zeros(n, dtype=float)
    for u in rev_order:
        best = 0.0
        for v in succ[u]:
            if bwd_rank[v] > best:
                best = bwd_rank[v]
        bwd_rank[u] = times[u] + best

    # Transitive descendant count (number of tasks that transitively depend on u)
    desc_count = np.zeros(n, dtype=int)
    # Process in reverse topological order: desc_count[u] = sum(1 + desc_count[v] for v in succ[u])
    # But that double-counts if multiple paths. Use a proper transitive closure count.
    # For small n (30), we can do BFS/DFS from each node.
    for u in range(n):
        visited = set()
        stack = list(succ[u])
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                stack.extend(succ[v])
        desc_count[u] = len(visited)

    # Number of direct predecessors (in-degree)
    indeg = np.array([len(p) for p in preds_list], dtype=float)

    # Number of direct successors (out-degree)
    outdeg = np.array([len(s) for s in succ], dtype=float)

    # Combine into a single priority score
    # Critical path weight: fwd_rank + bwd_rank - times (to avoid double counting)
    crit = fwd_rank + bwd_rank - times

    # Urgency: tasks with many transitive descendants are more "bottleneck"
    urgency = desc_count.astype(float)

    # Balance: prefer tasks that are on long paths AND unlock many successors
    # Use a weighted combination
    # Normalize each component to [0, 1] range for stable weighting
    def normalize(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.ones_like(x) * 0.5
        return (x - mn) / (mx - mn)

    n_crit = normalize(crit)
    n_time = normalize(times)
    n_urg = normalize(urgency)
    n_indeg = normalize(indeg)
    n_outdeg = normalize(outdeg)

    # Weighted combination:
    # - Critical path position (most important): 0.35
    # - Task duration: 0.25
    # - Transitive descendant urgency: 0.20
    # - In-degree (how constrained): 0.10
    # - Out-degree (how much it unlocks): 0.10
    score = (0.35 * n_crit + 0.25 * n_time + 0.20 * n_urg + 0.10 * n_indeg + 0.10 * n_outdeg)

    return score
# EVOLVE-BLOCK-END
