import numpy as np
import time

def priority(instance):
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    predecessors = instance["predecessors"]
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capancies"] if "capancies" in instance else instance["capacities"], dtype=float)

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
    
    # 3. Forward Pass: Compute Earliest Start (ES) and Earliest Finish (EF)
    ES = np.zeros(n)
    EF = np.zeros(n)
    for i in topo_order:
        max_pred_ef = 0.0
        for p in predecessors[i]:
            if EF[p] > max_pred_ef:
                max_pred_ef = EF[p]
        ES[i] = max_pred_ef
        EF[i] = ES[i] + durations[i]
        
    project_duration = np.max(EF)
    
    # 4. Backward Pass: Compute Latest Finish (LF) and Latest Start (LS)
    # Initialize LF for sink nodes (no successors) to project_duration
    LF = np.full(n, project_duration)
    LS = np.full(n, project_duration)
    
    for i in reversed(topo_order):
        if successors[i]:
            min_succ_ls = float('inf')
            for s in successors[i]:
                if LS[s] < min_succ_ls:
                    min_succ_ls = LS[s]
            LF[i] = min_succ_ls
        # If no successors, LF[i] remains project_duration
        LS[i] = LF[i] - durations[i]
        
    # 5. Compute Resource Intensity
    # RI_i = sum(demand_ij / capacity_j)
    # Handle zero capacities to avoid division errors
    safe_caps = np.where(capacities == 0, 1.0, capacities)
    resource_usage = np.sum(demands / safe_caps[np.newaxis, :], axis=1)
    
    # Normalize resource usage to [0, 1]
    min_ru = np.min(resource_usage)
    max_ru = np.max(resource_usage)
    ru_range = max_ru - min_ru
    if ru_range > 1e-10:
        resource_usage_norm = (resource_usage - min_ru) / ru_range
    else:
        resource_usage_norm = np.zeros(n)
        
    # 6. Calculate Priority Score
    # Urgency: Activities with smaller LF (earlier latest finish) are more critical.
    # We want high score for high urgency.
    # Urgency = MaxLF - LF
    max_lf = np.max(LF)
    urgency = max_lf - LF
    
    # Normalize urgency to [0, 1] for consistent scaling with resource usage
    min_urgency = np.min(urgency)
    max_urgency = np.max(urgency)
    urgency_range = max_urgency - min_urgency
    if urgency_range > 1e-10:
        urgency_norm = (urgency - min_urgency) / urgency_range
    else:
        urgency_norm = np.zeros(n)
        
    # Combine: Primarily urgency (LFT-based), secondarily resource intensity
    # Weight 0.8 for urgency, 0.2 for resource intensity
    scores = 0.8 * urgency_norm + 0.2 * resource_usage_norm
    
    # Optional: Add a tiny tie-breaker based on duration (longer first? shorter first?)
    # Usually, for RCPSP, if criticality is equal, shorter processing time (SPT) 
    # or higher resource intensity is preferred. We already have RI.
    # Let's stick to the weighted sum.
    
    return scores
