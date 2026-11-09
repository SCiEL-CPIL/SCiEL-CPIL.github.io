import numpy as np

def priority(instance):
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
    # Topological order (activities should be topologically sorted or we need to handle it)
    # Assume activities are in topological order (0 is first, etc.)
    # Actually, we should do a proper topological sort or assume the given order is topological
    # For safety, let's do forward pass assuming topological order
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
    # For each activity, compute max demand ratio across resources
    if n_res > 0:
        demand_ratios = demands / np.maximum(capacities, 1e-9)  # (n, n_res)
        max_demand_ratio = np.max(demand_ratios, axis=1)  # (n,)
    else:
        max_demand_ratio = np.zeros(n)
    
    # Normalize features
    max_float = np.max(total_float) if np.max(total_float) > 0 else 1.0
    max_tail = np.max(tail) if np.max(tail) > 0 else 1.0
    max_dur = np.max(durations) if np.max(durations) > 0 else 1.0
    max_ratio = np.max(max_demand_ratio) if np.max(max_demand_ratio) > 0 else 1.0
    
    # Priority: higher is more urgent
    # Low float -> high priority
    # High tail -> high priority  
    # High duration -> high priority (longer tasks are more impactful)
    # High resource demand -> high priority (scarce resource consumers)
    
    w_float = 0.5
    w_tail = 0.25
    w_dur = 0.15
    w_res = 0.10
    
    priority_scores = (
        w_float * (max_float - total_float) / max_float +
        w_tail * tail / max_tail +
        w_dur * durations / max_dur +
        w_res * max_demand_ratio / max_ratio
    )
    
    return priority_scores
