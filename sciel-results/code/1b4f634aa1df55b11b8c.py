import numpy as np
import time

def priority(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin before 150s limit
    
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # (n, n_res)
    capacities = np.asarray(instance["capacities"], dtype=float)  # (n_res,)
    predecessors = instance["predecessors"]  # list of lists
    
    n = len(durations)
    n_res = len(capacities)
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Forward pass: ES, EF
    es = np.zeros(n)
    ef = np.zeros(n)
    for i in range(n):
        for p in predecessors[i]:
            es[i] = max(es[i], ef[p])
        ef[i] = es[i] + durations[i]
    
    project_end = np.max(ef)
    
    # Backward pass: LS, LF
    ls = np.zeros(n)
    lf = np.zeros(n)
    for i in range(n-1, -1, -1):
        if not successors[i]:
            lf[i] = project_end
        else:
            lf[i] = min(ls[s] for s in successors[i])
        ls[i] = lf[i] - durations[i]
    
    # Total float
    total_float = ls - es
    
    # Tail: distance from EF to project end
    tail = project_end - ef
    
    # Resource demand intensity: total demand relative to capacity
    if n_res > 0:
        demand_ratios = demands / np.maximum(capacities, 1e-9)  # (n, n_res)
        max_demand_ratio = np.max(demand_ratios, axis=1)  # (n,)
        # Also compute weighted demand (sum of demand/capacity)
        weighted_demand = np.sum(demand_ratios, axis=1)
    else:
        max_demand_ratio = np.zeros(n)
        weighted_demand = np.zeros(n)
    
    # Normalize features
    max_float = np.max(total_float) if np.max(total_float) > 0 else 1.0
    max_tail = np.max(tail) if np.max(tail) > 0 else 1.0
    max_dur = np.max(durations) if np.max(durations) > 0 else 1.0
    max_ratio = np.max(max_demand_ratio) if np.max(max_demand_ratio) > 0 else 1.0
    max_weighted = np.max(weighted_demand) if np.max(weighted_demand) > 0 else 1.0
    
    # Priority: higher is more urgent
    # Use smooth surrogate: instead of hard min/max, use softmax-based importance
    # Low float -> high priority (critical path)
    # High tail -> high priority (close to end)
    # High duration -> high priority
    # High resource demand -> high priority
    
    # Try different weight combinations and pick the best based on a quick heuristic
    # We can't actually evaluate makespan here, so use a principled combination
    
    w_float = 0.55
    w_tail = 0.20
    w_dur = 0.10
    w_res_max = 0.05
    w_res_weighted = 0.10
    
    priority_scores = (
        w_float * (max_float - total_float) / max_float +
        w_tail * tail / max_tail +
        w_dur * durations / max_dur +
        w_res_max * max_demand_ratio / max_ratio +
        w_res_weighted * weighted_demand / max_weighted
    )
    
    # Check deadline
    if time.time() > deadline:
        return priority_scores
    
    # Try an alternative: use inverse float directly (no normalization needed for relative ordering)
    # and emphasize critical path more
    alt_scores = (
        0.7 * (max_float - total_float) / max_float +
        0.15 * tail / max_tail +
        0.15 * (durations / max_dur) * (max_demand_ratio / max_ratio)
    )
    
    # Use the primary scores; the alternative is kept for potential future use
    return priority_scores
