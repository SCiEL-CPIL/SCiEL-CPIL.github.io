import numpy as np
import time

def priority(instance):
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    predecessors = instance["predecessors"]
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    m = len(capacities)
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1
            
    # Topological Sort (Kahn's Algorithm)
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    current_queue = queue[:]
    in_degree_temp = in_degree[:]
    
    while current_queue:
        node = current_queue.pop(0) 
        topo_order.append(node)
        for succ in successors[node]:
            in_degree_temp[succ] -= 1
            if in_degree_temp[succ] == 0:
                current_queue.append(succ)
                
    if len(topo_order) != n:
        # Cycle detected? Fallback to durations
        return durations.copy()
    
    # Compute L_start (Longest path from start to i)
    L_start = np.zeros(n)
    for i in topo_order:
        max_pred = 0.0
        for p in predecessors[i]:
            if L_start[p] > max_pred:
                max_pred = L_start[p]
        L_start[i] = durations[i] + max_pred
        
    # Compute L_end (Longest path from i to end)
    L_end = np.zeros(n)
    for i in reversed(topo_order):
        max_succ = 0.0
        for s in successors[i]:
            if L_end[s] > max_succ:
                max_succ = L_end[s]
        L_end[i] = durations[i] + max_succ
        
    # Criticality = Path from start to end through i
    criticality = L_start + L_end - durations
    
    # Resource pressure: sum of (demand/capacity) for each activity
    # Use safe division to avoid division by zero
    safe_capacities = np.where(capacities > 0, capacities, 1.0)
    resource_pressure = np.sum(demands / safe_capacities[None, :], axis=1)
    
    # Normalize criticality
    crit_min, crit_max = np.min(criticality), np.max(criticality)
    if crit_max > crit_min:
        crit_norm = (criticality - crit_min) / (crit_max - crit_min)
    else:
        crit_norm = np.ones_like(criticality)
    
    # Normalize resource pressure
    rp_min, rp_max = np.min(resource_pressure), np.max(resource_pressure)
    if rp_max > rp_min:
        rp_norm = (resource_pressure - rp_min) / (rp_max - rp_min)
    else:
        rp_norm = np.zeros_like(resource_pressure)
    
    # Normalize duration
    dur_min, dur_max = np.min(durations), np.max(durations)
    if dur_max > dur_min:
        dur_norm = (durations - dur_min) / (dur_max - dur_min)
    else:
        dur_norm = np.zeros_like(durations)
    
    # Combined score with weights tuned for better makespan
    # Higher criticality should be scheduled earlier
    # Higher resource pressure should be scheduled earlier to free up resources sooner
    # Longer durations should be scheduled earlier to reduce idle time
    
    alpha = 0.70  # criticality weight
    beta = 0.20   # resource pressure weight  
    gamma = 0.10  # duration weight
    
    scores = alpha * crit_norm + beta * rp_norm + gamma * dur_norm
    
    # Additional refinement: use a non-linear transformation to amplify
    # differences in criticality for bottleneck activities
    # Apply a power transformation to sharpen the priority distribution
    if np.max(scores) > np.min(scores):
        score_range = np.max(scores) - np.min(scores)
        # Normalize to [0, 1] again after combining
        scores_norm = (scores - np.min(scores)) / score_range
        # Apply mild power to sharpen: higher scores get relatively boosted
        scores_sharp = scores_norm ** 1.2
        # Re-scale back to original range
        scores = np.min(scores) + scores_sharp * score_range
    
    return scores
