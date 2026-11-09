"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    """
    Compute priority scores for assembly tasks using a hybrid heuristic:
    1. Positional Weight (total downstream work including self)
    2. Immediate successor count (breadth of unlock)
    3. Task time (longer tasks first)
    
    Returns an array of priority scores (higher = more priority).
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    
    # Parse predecessors
    pred = instance["predecessors"]
    
    # Build successor lists
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in pred[i]:
            succ[p].append(i)
    
    # Compute in-degrees for topological sort
    in_degree = [0] * n
    for i in range(n):
        for p in pred[i]:
            in_degree[i] += 1
            
    # Topological sort (Kahn's algorithm)
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
            
    topo_order = []
    in_degree_copy = in_degree[:]
    while q:
        node = q.popleft()
        topo_order.append(node)
        for s in succ[node]:
            in_degree_copy[s] -= 1
            if in_degree_copy[s] == 0:
                q.append(s)
                
    if len(topo_order) != n:
        # Fallback for cycles (should not happen in valid ALB instances)
        return times.copy()
    
    # Calculate Positional Weights in reverse topological order
    # PW[i] = time[i] + sum(PW[s] for s in succ[i])
    pos_weight = np.zeros(n, dtype=float)
    
    for node in reversed(topo_order):
        total_succ_weight = 0.0
        for s in succ[node]:
            total_succ_weight += pos_weight[s]
        pos_weight[node] = times[node] + total_succ_weight
    
    # Number of immediate successors
    num_succ = np.array([len(succ[i]) for i in range(n)], dtype=float)
    
    # Normalize components
    max_pw = pos_weight.max() if pos_weight.max() > 0 else 1.0
    max_succ = num_succ.max() if num_succ.max() > 0 else 1.0
    max_time = times.max() if times.max() > 0 else 1.0
    
    # Combine: Positional Weight dominates, with small boosts for successors and time
    pw_norm = pos_weight / max_pw
    succ_norm = num_succ / max_succ
    time_norm = times / max_time
    
    # Priority: higher is better
    # Emphasize positional weight heavily, with secondary signals
    priority_scores = 0.7 * pw_norm + 0.2 * succ_norm + 0.1 * time_norm
    
    return priority_scores
# EVOLVE-BLOCK-END
