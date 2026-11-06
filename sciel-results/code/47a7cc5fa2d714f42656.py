import numpy as np
import time

def priority(instance):
    start_time = time.time()
    deadline = 120.0  # Safety margin
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    predecessors = instance["predecessors"]
    
    # Extract resource data if available
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    # 1. Build successor lists
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1
            
    # 2. Topological Sort (Kahn's Algorithm)
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
    
    # 3. Compute L_start (Longest path from start to i)
    L_start = np.zeros(n)
    for i in topo_order:
        max_pred = 0.0
        for p in predecessors[i]:
            if L_start[p] > max_pred:
                max_pred = L_start[p]
        L_start[i] = durations[i] + max_pred
        
    # 4. Compute L_end (Longest path from i to end)
    L_end = np.zeros(n)
    for i in reversed(topo_order):
        max_succ = 0.0
        for s in successors[i]:
            if L_end[s] > max_succ:
                max_succ = L_end[s]
        L_end[i] = durations[i] + max_succ
        
    # 5. Calculate base priority: critical path length through activity
    # Criticality = L_start[i] + L_end[i] - durations[i]
    criticality = L_start + L_end - durations
    
    # 6. Compute resource pressure: max demand-to-capacity ratio for each activity
    resource_pressure = np.zeros(n)
    if demands is not None and capacities is not None:
        demands_arr = np.asarray(demands)
        capacities_arr = np.asarray(capacities)
        # demands_arr shape: (n, num_resources)
        # capacities_arr shape: (num_resources,)
        if demands_arr.ndim == 2 and capacities_arr.ndim == 1:
            # For each activity, compute max(demand/capacity) across resources
            # Avoid division by zero
            safe_caps = np.where(capacities_arr > 0, capacities_arr, 1.0)
            ratios = demands_arr / safe_caps[np.newaxis, :]
            resource_pressure = np.max(ratios, axis=1)
    
    # 7. Combine scores:
    # Primary: criticality (higher = more urgent)
    # Secondary: resource pressure (higher = more resource-intensive, should schedule early)
    # Tertiary: duration (longer tasks benefit from earlier start)
    
    # Normalize each component to [0, 1] for stable weighting
    crit_min, crit_max = np.min(criticality), np.max(criticality)
    if crit_max > crit_min:
        crit_norm = (criticality - crit_min) / (crit_max - crit_min)
    else:
        crit_norm = np.ones(n)
    
    rp_min, rp_max = np.min(resource_pressure), np.max(resource_pressure)
    if rp_max > rp_min:
        rp_norm = (resource_pressure - rp_min) / (rp_max - rp_min)
    else:
        rp_norm = np.zeros(n)
    
    dur_min, dur_max = np.min(durations), np.max(durations)
    if dur_max > dur_min:
        dur_norm = (durations - dur_min) / (dur_max - dur_min)
    else:
        dur_norm = np.zeros(n)
    
    # Weighted combination: criticality dominates, then resource pressure, then duration
    weights = (0.7, 0.2, 0.1)
    scores = weights[0] * crit_norm + weights[1] * rp_norm + weights[2] * dur_norm
    
    return scores
