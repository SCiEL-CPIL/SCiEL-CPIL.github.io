import numpy as np
import time
import hashlib

# Global cache for CPM data to avoid recomputation if possible
# Key: hash of (durations, predecessors)
_cpm_cache = {}

def _get_cpm_data(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance["predecessors"]
    
    # Create a unique key for this instance structure
    # Hashing the list of predecessors and durations
    pred_str = str(predecessors)
    dur_str = str(durations)
    key = hashlib.md5((pred_str + dur_str).encode()).hexdigest()
    
    if key in _cpm_cache:
        return _cpm_cache[key]
        
    n = len(durations)
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    # Topological Sort
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    in_degree_temp = in_degree[:]
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for succ in successors[node]:
            in_degree_temp[succ] -= 1
            if in_degree_temp[succ] == 0:
                queue.append(succ)
                
    if len(topo_order) != n:
        # Fallback
        tf = np.zeros(n)
        _cpm_cache[key] = (tf, durations)
        return tf, durations

    # Forward Pass (Earliest Start/Finish)
    EF = np.zeros(n) # Earliest Finish
    for i in topo_order:
        max_pred_ef = 0.0
        for p in predecessors[i]:
            if EF[p] > max_pred_ef:
                max_pred_ef = EF[p]
        EF[i] = max_pred_ef + durations[i]
        
    project_duration = np.max(EF)
    
    # Backward Pass (Latest Start/Finish)
    LF = np.zeros(n) # Latest Finish
    # Initialize LF for activities with no successors to project_duration
    for i in range(n):
        if not successors[i]:
            LF[i] = project_duration
    
    for i in reversed(topo_order):
        min_succ_lf = project_duration
        for s in successors[i]:
            if LF[s] < min_succ_lf:
                min_succ_lf = LF[s]
        LF[i] = min_succ_lf - durations[i]
        
    # Total Float
    TF = LF - (EF - durations) # LF - ES
    
    _cpm_cache[key] = (TF, durations, successors, predecessors)
    return TF, durations, successors, predecessors

def priority(instance):
    # Check for dynamic state
    # Common keys in such simulations
    current_time = None
    avail_res = None
    
    # Try to find current time
    if "current_time" in instance:
        current_time = instance["current_time"]
    elif "time" in instance:
        current_time = instance["time"]
    elif "t" in instance:
        current_time = instance["t"]
        
    # Try to find resource availability
    if "resource_availability" in instance:
        avail_res = np.asarray(instance["resource_availability"], dtype=float)
    elif "avail_res" in instance:
        avail_res = np.asarray(instance["avail_res"], dtype=float)
    elif "remaining_capacity" in instance:
        avail_res = np.asarray(instance["remaining_capacity"], dtype=float)

    TF, durations, successors, predecessors = _get_cpm_data(instance)
    
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    
    # Static Resource Usage (Demand / Capacity)
    # Avoid division by zero
    safe_cap = np.where(capacities > 0, capacities, 1.0)
    static_res_usage = np.sum(demands / safe_cap[np.newaxis, :], axis=1)
    
    if current_time is not None and avail_res is not None:
        # Dynamic Heuristic
        # Calculate pressure based on current availability
        # If avail_res is low, demand is more critical
        # Pressure = sum( demand / (avail + demand) )
        # Or simpler: sum( demand / (avail + epsilon) )
        
        # Ensure avail_res is not zero to avoid inf
        safe_avail = np.where(avail_res > 0, avail_res, 1e-6)
        
        # Dynamic resource load
        dynamic_load = np.sum(demands / safe_avail[np.newaxis, :], axis=1)
        
        # Normalize dynamic load to be comparable to TF?
        # TF can be large. dynamic_load can be large.
        # Let's use a weighted sum: Score = -TF + alpha * dynamic_load
        # Or multiplicative: Score = -TF * (1 + dynamic_load)
        
        # Let's try: Higher priority for lower TF and higher dynamic load.
        # Since we return "priority" where higher is better:
        # We want to maximize (-TF + beta * dynamic_load)
        
        # Beta tuning: 
        # If dynamic_load is huge, it dominates.
        # Let's scale dynamic_load.
        # Max possible dynamic load is sum(demands/capacities) if avail=0.
        # Let's just use a small beta to act as a tie-breaker or mild boost.
        
        beta = 0.5
        scores = -TF + beta * dynamic_load
        
    else:
        # Static Heuristic
        # Use static resource usage as a proxy for contention
        # Score = -TF + alpha * static_res_usage
        # Current program used: 1.0 * (L_start+L_end-d) which is ~ TF (with sign flip?)
        # Wait, L_start + L_end - d is NOT TF.
        # TF = LF - ES.
        # L_start = ES + d.
        # L_end = LF + d? No, L_end in previous code was "longest path from i to end".
        # PathLen(i) = d_i + max(PathLen(succ)).
        # LF_i = ProjectDur - PathLen(i) + d_i?
        # Actually, Criticality in previous code:
        # L_start[i] = ES_i + d_i = EF_i
        # L_end[i] = d_i + max(L_end[s]) = d_i + (LF_s - d_s ...)?
        # Let's stick to standard TF computed above.
        
        # TF is lower for critical tasks. We want to prioritize them.
        # So we return -TF.
        # Add resource weight.
        
        alpha = 0.2
        scores = -TF + alpha * static_res_usage
        
    return scores
