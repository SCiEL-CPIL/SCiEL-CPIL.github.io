"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _compute_priorities(instance):
    """
    Compute a composite priority for each task in an assembly line instance.
    
    The priority combines:
    1. Longest path to end (successor rank / critical path weight)
    2. Task duration
    3. Number of successors (breadth of downstream impact)
    
    Returns an array of scores, one per task, where higher = more urgent.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    
    # Parse predecessors
    preds = instance.get("predecessors", [])
    # preds is typically a list where preds[i] is a list of predecessor indices for task i
    # Or it could be a dict. Let's handle common formats.
    
    # Build successor adjacency
    succ = [[] for _ in range(n)]
    in_degree = [0] * n
    
    if isinstance(preds, dict):
        for i, plist in preds.items():
            for p in plist:
                succ[p].append(i)
                in_degree[i] += 1
    elif isinstance(preds, (list, tuple)):
        for i, plist in enumerate(preds):
            for p in plist:
                succ[p].append(i)
                in_degree[i] += 1
    else:
        # If preds is None or empty, no precedence constraints
        pass
    
    # Compute longest path to end (successor rank) using topological sort
    # longest_to_end[i] = times[i] + max(longest_to_end[j] for j in succ[i])
    # We process in reverse topological order
    
    # Topological sort using Kahn's algorithm
    from collections import deque
    
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succ[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If there's a cycle, fall back to just duration
    if len(topo_order) < n:
        return times.copy()
    
    # Reverse topological order for computing longest path to end
    longest_to_end = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if succ[u]:
            longest_to_end[u] = times[u] + np.max([longest_to_end[v] for v in succ[u]])
        else:
            longest_to_end[u] = times[u]
    
    # Count number of successors (immediate)
    num_succ = np.array([len(succ[i]) for i in range(n)], dtype=float)
    
    # Compute work remaining (sum of times of task and all its successors)
    # N is small (30), so O(N^2) DFS/BFS is fine
    work = np.zeros(n, dtype=float)
    for i in range(n):
        # BFS/DFS to find all reachable successors
        visited = set([i])
        stack = [i]
        total_work = 0
        while stack:
            node = stack.pop()
            total_work += times[node]
            for s in succ[node]:
                if s not in visited:
                    visited.add(s)
                    stack.append(s)
        work[i] = total_work
    
    # Compute bottleneck density
    # Ratio of task time to minimum stations required for its downstream work
    cycle_time = instance.get("cycle_time", 40.0)
    if cycle_time <= 0:
        cycle_time = 40.0
        
    min_stations = np.zeros(n, dtype=float)
    for i in range(n):
        min_stations[i] = max(1.0, math.ceil(work[i] / cycle_time))
        
    bottleneck = times / min_stations
    
    # Normalize components to [0, 1] range for combination
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    norm_lte = normalize(longest_to_end)
    norm_bottleneck = normalize(bottleneck)
    norm_succ = normalize(num_succ)
    
    # Weighted combination
    # Heavily weight longest path to end, then bottleneck density, then breadth
    # Bottleneck density replaces raw time as it accounts for remaining load
    priority_scores = 1.0 * norm_lte + 0.5 * norm_bottleneck + 0.1 * norm_succ
    
    return priority_scores

def priority(instance):
    """
    Score each assembly task. Stations are filled with the highest-priority
    eligible task that fits the cycle time.
    
    Returns one score per task (higher = more urgent).
    """
    start_time = time.time()
    
    try:
        scores = _compute_priorities(instance)
    except Exception:
        # Fallback to duration-based priority
        times = np.asarray(instance["times"], dtype=float)
        scores = times.copy()
    
    # Deadline guard: if we've used too much time, return what we have
    elapsed = time.time() - start_time
    if elapsed > 10:  # 10 second safety margin
        pass  # We already have a result, just return it
    
    return scores
# EVOLVE-BLOCK-END
