import numpy as np
import time
from collections import deque

def priority(instance):
    t_start = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    predecessors = instance["predecessors"]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Build successor map: succ[i] = list of tasks that depend on i
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            succ[p].append(i)
    
    # Topological sort (Kahn's algorithm)
    in_degree = [len(predecessors[i]) for i in range(n)]
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in succ[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # If not all nodes, there's a cycle - fallback
    if len(topo_order) != n:
        return times
    
    # Compute longest path from each node to a terminal (reverse topological order)
    # lp[i] = times[i] + max(lp[s] for s in succ[i]) if succ[i] else times[i]
    lp = [0.0] * n
    for node in reversed(topo_order):
        if succ[node]:
            lp[node] = times[node] + max(lp[s] for s in succ[node])
        else:
            lp[node] = times[node]
    
    # Compute weighted precedence: sum of times of task and all transitive successors
    # wp[i] = times[i] + sum(wp[s] for s in succ[i])
    wp = [0.0] * n
    for node in reversed(topo_order):
        w = times[node]
        for s in succ[node]:
            w += wp[s]
        wp[node] = w
    
    # Compute number of immediate successors
    num_succ = [len(succ[i]) for i in range(n)]
    
    # Compute depth: number of levels from node to deepest terminal
    depth = [0] * n
    for node in reversed(topo_order):
        if succ[node]:
            depth[node] = 1 + max(depth[s] for s in succ[node])
        else:
            depth[node] = 0
    
    # Compute number of immediate predecessors
    num_pred = [len(predecessors[i]) for i in range(n)]
    
    # Structural change: Use a two-tier priority scheme
    # Tier 1: Hard constraint - tasks with no unscheduled predecessors get boosted
    # Tier 2: Soft ranking based on combined heuristics
    # This mimics the greedy nature of the evaluator more closely
    
    # Normalize each component to [0, 1] range
    lp_arr = np.array(lp)
    wp_arr = np.array(wp)
    succ_arr = np.array(num_succ, dtype=float)
    depth_arr = np.array(depth, dtype=float)
    time_arr = times
    pred_arr = np.array(num_pred, dtype=float)
    
    # Use rank-based normalization for robustness
    lp_ranks = np.argsort(np.argsort(lp_arr))
    wp_ranks = np.argsort(np.argsort(wp_arr))
    succ_ranks = np.argsort(np.argsort(succ_arr))
    depth_ranks = np.argsort(np.argsort(depth_arr))
    time_ranks = np.argsort(np.argsort(time_arr))
    
    lp_norm = lp_ranks / max(1, n - 1)
    wp_norm = wp_ranks / max(1, n - 1)
    succ_norm = succ_ranks / max(1, n - 1)
    depth_norm = depth_ranks / max(1, n - 1)
    time_norm = time_ranks / max(1, n - 1)
    
    # Base priority score: weighted combination favoring LP and WP
    base_score = (0.55 * lp_norm + 
                  0.25 * wp_norm + 
                  0.10 * succ_norm + 
                  0.05 * depth_norm + 
                  0.05 * time_norm)
    
    # Structural enhancement: Boost tasks that are "ready" (no predecessors)
    # This aligns with the greedy filling strategy
    ready_boost = np.zeros(n)
    for i in range(n):
        if num_pred[i] == 0:
            ready_boost[i] = 0.15  # Significant boost for initially available tasks
    
    # Additional structural feature: Penalty for tasks that block many others
    # If a task has many successors, scheduling it early unlocks more options
    unlock_factor = succ_norm * 0.10
    
    # Combine all components
    priority_scores = base_score + ready_boost + unlock_factor
    
    # Check time budget
    if time.time() - t_start > deadline:
        return priority_scores
    
    # Final polish: ensure no ties by adding tiny epsilon based on topological order
    # This helps break ties deterministically in favor of earlier topological order
    eps = 1e-10
    topo_rank = np.zeros(n)
    for idx, node in enumerate(topo_order):
        topo_rank[node] = idx
    
    # Earlier in topological order gets slightly higher priority (tie-breaker)
    topo_norm = (n - 1 - topo_rank) / max(1, n - 1)
    priority_scores += eps * topo_norm
    
    return priority_scores
