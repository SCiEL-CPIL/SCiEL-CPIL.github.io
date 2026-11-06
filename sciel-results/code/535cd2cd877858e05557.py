import numpy as np
import time

def priority(instance):
    """
    Returns a priority score for each activity to be used in a Serial SGS.
    
    Strategy: 
    1. Compute Critical Path Length (CPL) and Earliest Start Times (EST).
    2. Compute a 'Bottleneck Pressure' heuristic.
       This heuristic weights the CPL by the relative scarcity of resources 
       consumed by the activity. It approximates the marginal impact of 
       starting the activity on the overall resource contention.
       Formula: P_i = CPL_i * (1 + sum_r (d_{ir} / max(C_r - d_{ir}, 1)))
       Note: We use a safe denominator to avoid division by zero if demand 
       equals capacity, though typically d < C.
    3. Incorporate 'Slack' penalty: Activities with zero slack (critical) 
       get a boost.
    4. Return the composite priority vector.
    """
    
    # 1. Extract and process instance data
    N = len(instance["durations"])
    durations = np.array(instance["durations"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)  # Shape (N, R)
    capacities = np.array(instance["capacities"], dtype=float) # Shape (R,)
    R = len(capacities)
    
    # Handle predecessors
    preds = instance.get("predecessors", [[] for _ in range(N)])
    if isinstance(preds, np.ndarray):
        if preds.ndim == 2:
             preds = [np.where(preds[i] > 0)[0].tolist() for i in range(N)]
        else:
             preds = [[] for _ in range(N)]
    elif isinstance(preds, list):
         normalized_preds = []
         for p in preds:
             if isinstance(p, (list, tuple, np.ndarray)):
                 normalized_preds.append(list(map(int, p)))
             else:
                 normalized_preds.append([])
         preds = normalized_preds
    else:
         preds = [[] for _ in range(N)]

    # Build successors
    succs = [[] for _ in range(N)]
    for i in range(N):
        for p in preds[i]:
            succs[p].append(i)
            
    # 2. Topological Sort
    in_degree = [len(p) for p in preds]
    queue = [i for i in range(N) if in_degree[i] == 0]
    topo_order = []
    indeg_temp = in_degree[:]
    q = queue[:]
    while q:
        u = q.pop(0)
        topo_order.append(u)
        for v in succs[u]:
            indeg_temp[v] -= 1
            if indeg_temp[v] == 0:
                q.append(v)
                
    # 3. Compute Critical Path Length (CPL) backwards
    cpl = np.zeros(N)
    for i in range(len(topo_order) - 1, -1, -1):
        u = topo_order[i]
        max_succ_cpl = 0
        for v in succs[u]:
            if cpl[v] > max_succ_cpl:
                max_succ_cpl = cpl[v]
        cpl[u] = durations[u] + max_succ_cpl
        
    # 4. Compute Earliest Start Times (EST) forwards
    est = np.zeros(N)
    for i in range(len(topo_order)):
        u = topo_order[i]
        max_pred_est = 0
        for p in preds[u]:
            if est[p] + durations[p] > max_pred_est:
                max_pred_est = est[p] + durations[p]
        est[u] = max_pred_est
        
    # 5. Compute Total Slack
    # LST (Latest Start Time) is calculated as CPL - Duration - Successor CPL
    # Actually, Slack = LST - EST. 
    # LST_i = min(LST_succ - duration_succ) for all succs.
    # Alternatively, Slack_i = (LFT_i - LST_i) is 0 for duration, so Slack = LST - EST.
    # Let's compute LST backwards.
    lst = np.zeros(N)
    # Initialize LST with infinity, but we can derive it from CPL
    # LFT_i = CPL_i. LST_i = LFT_i - duration_i.
    # Wait, CPL_i is the length of the longest path from i to end.
    # So LFT_i (Latest Finish) relative to project end?
    # Standard CPM:
    # EST_i = max(EST_p + d_p)
    # EFT_i = EST_i + d_i
    # LFT_i = min(LST_s - d_s) for successors s. LFT_sink = EFT_sink (Project Duration).
    # LST_i = LFT_i - d_i.
    # Slack_i = LST_i - EST_i.
    
    project_duration = np.max(est + durations)
    
    # Calculate LFT backwards
    lft = np.full(N, project_duration)
    for i in range(len(topo_order) - 1, -1, -1):
        u = topo_order[i]
        if succs[u]:
            min_lft_succ = float('inf')
            for v in succs[u]:
                if lft[v] - durations[v] < min_lft_succ:
                    min_lft_succ = lft[v] - durations[v]
            lft[u] = min_lft_succ
        # If no successors, lft[u] remains project_duration (which is correct for sinks)
        
    lst = lft - durations
    slack = lst - est
    
    # 6. Compute Bottleneck Pressure Heuristic
    # We want to prioritize activities that are:
    # a) On the critical path (high CPL, low Slack)
    # b) Consume resources that are close to their limit (high demand relative to capacity)
    
    # Calculate resource intensity factor for each activity
    # Factor_r = d_ir / (C_r - d_ir) if C_r > d_ir else large_number
    # This penalizes activities that take a large chunk of a scarce resource.
    
    # To avoid division by zero or negative numbers (if d > C, which shouldn't happen in valid instances, but safe to handle):
    # effective_cap = max(C_r - d_ir, 1.0)
    
    pressure = np.zeros(N)
    for r in range(R):
        cap_r = capacities[r]
        if cap_r <= 0:
            continue
        # d_ir is demands[:, r]
        # We calculate the ratio of demand to remaining capacity if this activity started now?
        # No, this is a static heuristic. We use the ratio of demand to total capacity, 
        # but amplified if the demand is a significant fraction of the capacity.
        # Let's use: ratio = d / (C - d)
        # If d is small, ratio is small. If d is close to C, ratio is huge.
        
        d_col = demands[:, r]
        remaining_cap = cap_r - d_col
        # Safety: if remaining_cap <= 0, set to 1.0 to keep ratio high but finite
        safe_remaining = np.where(remaining_cap > 0, remaining_cap, 1.0)
        ratio = d_col / safe_remaining
        
        # Add to pressure
        pressure += ratio
        
    # Normalize pressure to [0, 1] range for better scaling?
    # Or just use it directly.
    # Let's normalize by max pressure to keep numbers manageable.
    max_pressure = np.max(pressure)
    if max_pressure > 0:
        pressure = pressure / max_pressure
        
    # 7. Composite Priority
    # Base: CPL
    # Modifier: (1 + pressure) -> boosts resource-heavy activities
    # Modifier: (1 - slack/max_slack) -> boosts tight activities
    
    max_slack = np.max(slack)
    if max_slack > 0:
        slack_factor = 1.0 - (slack / max_slack)
    else:
        slack_factor = np.ones(N)
        
    # Priority = CPL * (1 + 2 * pressure) * (1 + slack_factor)
    # The factor 2 is a heuristic weight to emphasize resource contention.
    # We can tune this, but 2 is a reasonable starting point for "Bottleneck Pressure".
    
    priorities = cpl * (1.0 + 2.0 * pressure) * (1.0 + slack_factor)
    
    # Ensure priorities are distinct enough to break ties deterministically if needed,
    # though the SGS handles ties by index usually.
    # Add a tiny epsilon based on index to break ties in favor of lower indices if values are identical?
    # Not strictly necessary, but good for reproducibility.
    # priorities += np.arange(N) * 1e-6
    
    return priorities
