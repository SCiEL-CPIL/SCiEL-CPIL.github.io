"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    preds = instance.get("predecessors", [list() for _ in range(n)])
    
    # Build successor adjacency
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
    
    # Topological order (Kahn's algorithm)
    in_deg = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            in_deg[i] += 1
    
    from collections import deque
    q = deque()
    for i in range(n):
        if in_deg[i] == 0:
            q.append(i)
    
    topo = []
    indeg_work = in_deg.copy()
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succs[u]:
            indeg_work[v] -= 1
            if indeg_work[v] == 0:
                q.append(v)
    
    # If topological sort failed (cycle), fall back to times
    if len(topo) < n:
        return times.copy()
    
    # Longest path from task i to any end (including i's own time)
    # Process in reverse topological order
    longest_path = np.zeros(n, dtype=float)
    for i in reversed(range(len(topo))):
        u = topo[i]
        lp = times[u]
        for v in succs[u]:
            if longest_path[v] > lp:
                lp = longest_path[v]
        longest_path[u] = lp
    
    # Total downstream work: sum of times of all successors (not including self)
    # Process in reverse topological order
    downstream_work = np.zeros(n, dtype=float)
    for i in reversed(range(len(topo))):
        u = topo[i]
        dw = 0.0
        for v in succs[u]:
            dw += times[v] + downstream_work[v]
        downstream_work[u] = dw
    
    # Also compute number of successors (direct + indirect) for tie-breaking
    # depth: number of tasks in the longest chain after u
    depth = np.zeros(n, dtype=float)
    for i in reversed(range(len(topo))):
        u = topo[i]
        d = 0
        for v in succs[u]:
            if 1 + depth[v] > d:
                d = 1 + depth[v]
        depth[u] = d
    
    # Combine: primary = longest_path, secondary = downstream_work, tertiary = depth
    # Normalize each to [0,1] range for fair combination
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    lp_norm = normalize(longest_path)
    dw_norm = normalize(downstream_work)
    depth_norm = normalize(depth)
    
    # Weighted combination: emphasize critical path most, then downstream work
    score = 0.5 * lp_norm + 0.35 * dw_norm + 0.15 * depth_norm
    
    return score
# EVOLVE-BLOCK-END
