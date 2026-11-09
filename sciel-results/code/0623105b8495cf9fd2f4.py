"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _topo_sort(times, preds):
    """Return a topological ordering of tasks."""
    n = len(times)
    # Build adjacency and in-degree
    adj = [[] for _ in range(n)]
    indeg = [0] * n
    for i in range(n):
        for p in preds[i]:
            adj[p].append(i)
            indeg[i] += 1
    
    from collections import deque
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order


def _compute_ranking(instance):
    """
    Compute a priority score for each task using a combination of:
    - Longest path to end (successor rank)
    - Task duration
    - Number of successors (successor count)
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    ct = instance["cycle_time"]
    
    # Build successor lists
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
    
    # Topological order
    order = _topo_sort(times, preds)
    
    # Compute longest path from each node to any sink (successor rank)
    # rank[i] = times[i] + max(rank[j] for j in succs[i]) if succs[i] else times[i]
    rank = np.zeros(n, dtype=float)
    for i in reversed(order):
        if succs[i]:
            rank[i] = times[i] + max(rank[j] for j in succs[i])
        else:
            rank[i] = times[i]
    
    # Count successors (transitive)
    succ_count = np.zeros(n, dtype=int)
    for i in reversed(order):
        if succs[i]:
            succ_count[i] = 1 + max(succ_count[j] for j in succs[i])
        else:
            succ_count[i] = 0
    
    # Combine: primarily rank, tie-break by duration, then successor count
    # Normalize to [0, 1] range for each component
    max_rank = rank.max() if rank.max() > 0 else 1.0
    max_time = times.max() if times.max() > 0 else 1.0
    max_succ = succ_count.max() if succ_count.max() > 0 else 1.0
    
    # Weighted combination: rank is most important
    score = 0.7 * (rank / max_rank) + 0.2 * (times / max_time) + 0.1 * (succ_count / max_succ)
    
    return score


def priority(instance):
    """
    Score each assembly task for priority.
    Higher score = higher priority (placed first).
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    if n == 0:
        return np.array([])
    
    # Compute the ranking
    score = _compute_ranking(instance)
    
    # Add a tiny deterministic noise to break ties consistently
    # based on task index to ensure stable ordering
    score = score + np.arange(n) * 1e-10
    
    return score
# EVOLVE-BLOCK-END
