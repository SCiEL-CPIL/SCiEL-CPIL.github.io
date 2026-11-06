"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_successors(instance):
    times = np.asarray(instance["times"], dtype=float)
    preds = instance.get("predecessors", [])
    n = len(times)
    succ = [[] for _ in range(n)]
    for i, ps in enumerate(preds):
        for p in ps:
            succ[p].append(i)
    return succ


def _compute_rank(instance, succ, times):
    # Rank: longest path from task to end (in reverse topological order)
    n = len(times)
    # We need topological order. Use Kahn's algorithm.
    preds = instance.get("predecessors", [])
    in_deg = np.zeros(n, dtype=int)
    for i, ps in enumerate(preds):
        for p in ps:
            in_deg[i] += 1
    
    from collections import deque
    q = deque([i for i in range(n) if in_deg[i] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succ[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                q.append(v)
    
    # Compute rank (longest path to sink) in reverse topo order
    rank = np.zeros(n, dtype=float)
    for u in reversed(topo):
        max_succ = 0.0
        for v in succ[u]:
            max_succ = max(max_succ, rank[v])
        rank[u] = times[u] + max_succ
    return rank


def _compute_rank2(instance, succ, times):
    # Alternative rank: consider also the number of successors
    n = len(times)
    preds = instance.get("predecessors", [])
    in_deg = np.zeros(n, dtype=int)
    for i, ps in enumerate(preds):
        for p in ps:
            in_deg[i] += 1
    
    from collections import deque
    q = deque([i for i in range(n) if in_deg[i] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succ[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                q.append(v)
    
    rank = np.zeros(n, dtype=float)
    for u in reversed(topo):
        max_succ = 0.0
        for v in succ[u]:
            max_succ = max(max_succ, rank[v])
        rank[u] = times[u] + max_succ
    return rank


def priority(instance):
    start_time = time.time()
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance.get("cycle_time", 40)
    
    succ = _compute_successors(instance)
    rank = _compute_rank(instance, succ, times)
    
    # Priority: primarily rank (longest path to end), tie-break by time descending
    # This is a standard heuristic for ALB
    # Higher priority = should be scheduled earlier
    # We want tasks that are "critical" (long path to end) to be scheduled first
    
    # Combine: rank is the primary factor
    # Add a small bonus for larger tasks to break ties
    priority_scores = rank + 0.01 * times
    
    return np.asarray(priority_scores, dtype=float)
# EVOLVE-BLOCK-END
