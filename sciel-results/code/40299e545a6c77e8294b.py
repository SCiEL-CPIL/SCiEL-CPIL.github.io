"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np
from collections import deque


def _compute_metrics(times, preds):
    """
    Computes:
    - BL (Bottom Level / Longest Path to sink)
    - SST (Sum of Successor Times + Self)
    - NS (Number of Immediate Successors)
    - NS_total (Number of All Successors) - optional, maybe too expensive? 
      For n=30, it's cheap.
    """
    n = len(times)
    succs = [[] for _ in range(n)]
    indeg = [0] * n
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
            indeg[i] += 1

    # Topological sort (Kahn's)
    q = deque([i for i in range(n) if indeg[i] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(topo) != n:
        topo = list(range(n))

    # Reverse topological order for DP
    rev_topo = reversed(topo)

    # 1. Bottom Level (Longest Path to sink, including self)
    bl = np.zeros(n, dtype=float)
    for u in rev_topo:
        best = 0.0
        for v in succs[u]:
            if bl[v] > best:
                best = bl[v]
        bl[u] = times[u] + best

    # 2. Sum of Successor Times (including self)
    # SST[u] = times[u] + sum(SST[v] for v in succs[u])? 
    # No, SST usually means sum of processing times of all successors.
    # Let's compute "Total Work Downstream" TWD[u] = times[u] + sum(TWD[v] for v in succs[u])
    # This counts each task exactly once in the subtree.
    twd = np.zeros(n, dtype=float)
    for u in rev_topo:
        total = times[u]
        for v in succs[u]:
            total += twd[v]
        twd[u] = total

    # 3. Number of Immediate Successors
    ns_immediate = np.array([len(s) for s in succs], dtype=float)

    # 4. Number of Total Successors (optional, might be redundant with TWD)
    # ns_total[u] = 1 + sum(ns_total[v] for v in succs[u]) if we count self?
    # Let's stick to immediate for simplicity, TWD captures the "size" better.

    return bl, twd, ns_immediate


def priority(instance):
    t0 = _time.time()
    DEADLINE = 140.0

    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)

    bl, twd, ns_immediate = _compute_metrics(times, preds)
    ct = float(instance.get("cycle_time", 40.0))

    # Heuristic 1: Critical Path Density
    # High time, low slack (BL close to CT) -> High Priority
    # Slack = CT - BL
    # If BL > CT, slack is negative. We want to prioritize these heavily.
    # Using a soft floor: slack_safe = max(CT - BL, epsilon)
    # Density = time / slack_safe
    eps = 0.1
    slack = ct - bl
    slack_safe = np.maximum(slack, eps)
    density = times / slack_safe

    # Heuristic 2: Total Work Downstream
    # High total work -> High Priority
    # Normalize by CT to make it scale-invariant? Or just max-normalize.
    # Let's use max-normalization for all components.
    
    # Heuristic 3: Successor Count
    # More successors -> more unblocking -> High Priority
    
    # Normalization
    def normalize(arr):
        arr = np.asarray(arr, dtype=float)
        if len(arr) == 0:
            return arr
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val < 1e-10:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    # We want to keep the "urgency" of density high.
    # Density can be very large if slack is small.
    # Normalizing density to [0,1] might compress the difference between "critical" and "very critical".
    # However, the greedy algorithm only needs the *order*.
    # So normalization is safe and helps with weighting.
    
    density_norm = normalize(density)
    twd_norm = normalize(twd)
    ns_norm = normalize(ns_immediate)
    
    # Weighting
    # Density is the most specific to the "fits in cycle time" constraint.
    # TWD is the classic "work content" heuristic.
    # NS is a tie-breaker.
    
    w_density = 1.0
    w_twd = 0.5
    w_ns = 0.1
    
    scores = w_density * density_norm + w_twd * twd_norm + w_ns * ns_norm
    
    # Scale to a reasonable magnitude (optional, but keeps numbers stable)
    # The relative order is what matters.
    
    if _time.time() - t0 > DEADLINE:
        return times

    return scores
# EVOLVE-BLOCK-END
