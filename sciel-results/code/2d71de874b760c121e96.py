"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def priority(instance):
    start_time = time.time()
    deadline = start_time + 10.0

    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)

    # Build successors from predecessors
    successors = [[] for _ in range(n)]
    for i, preds in enumerate(predecessors):
        for p in preds:
            successors[p].append(i)

    # Compute in-degree for topological sort
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1

    # Topological sort (Kahn's algorithm)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    topo_depth = np.zeros(n)
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for s in successors[u]:
            topo_depth[s] = max(topo_depth[s], topo_depth[u] + 1)
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)

    # Compute bottom-level in reverse topological order
    bl = np.zeros(n)
    for u in reversed(topo_order):
        if len(successors[u]) == 0:
            bl[u] = times[u]
        else:
            max_succ_bl = max(bl[s] for s in successors[u])
            bl[u] = times[u] + max_succ_bl

    # Compute successor count
    succ_count = np.array([len(succ) for succ in successors], dtype=float)

    # Normalize each component to [0, 1]
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    bl_norm = normalize(bl)
    succ_norm = normalize(succ_count)
    depth_norm = normalize(topo_depth)

    # Combine: prioritize high bottom-level, high successor count, low depth (early)
    priority_scores = 1.0 * bl_norm + 0.5 * succ_norm + 0.3 * (1.0 - depth_norm)

    return priority_scores
# EVOLVE-BLOCK-END
