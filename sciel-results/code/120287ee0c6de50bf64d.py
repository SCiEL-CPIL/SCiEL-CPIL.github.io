import numpy as np

def priority(instance):
    n = len(instance["durations"])
    if n == 0:
        return np.array([])
        
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]
    
    # Ensure demands is 2D
    if demands.ndim == 1:
        demands = demands.reshape(-1, 1)
        
    # Build successors
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    
    for i in range(n):
        preds = predecessors[i]
        in_degree[i] = len(preds)
        for p in preds:
            successors[p].append(i)
            
    # Topological Sort (Kahn's)
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    
    while queue:
        # Pop from end for efficiency, order doesn't matter for CPM
        u = queue.pop()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(topo_order) != n:
        # Fallback if cycle
        return durations
        
    # Forward Pass
    ES = np.zeros(n)
    EF = np.zeros(n)
    for u in topo_order:
        for v in successors[u]:
            if ES[v] < EF[u]:
                ES[v] = EF[u]
        EF[u] = ES[u] + durations[u]
        
    project_duration = np.max(EF) if n > 0 else 0
    
    # Backward Pass
    LF = np.full(n, project_duration)
    LS = np.full(n, project_duration)
    
    for u in reversed(topo_order):
        if not successors[u]:
            LF[u] = project_duration
        else:
            # Find min LS of successors
            min_ls = project_duration
            for v in successors[u]:
                if LS[v] < min_ls:
                    min_ls = LS[v]
            LF[u] = min_ls
        LS[u] = LF[u] - durations[u]
        
    # Total Float
    TF = LS - ES
    
    # Critical Path Length from each node (W)
    W = np.zeros(n)
    for u in reversed(topo_order):
        if not successors[u]:
            W[u] = durations[u]
        else:
            max_w = 0.0
            for v in successors[u]:
                if W[v] > max_w:
                    max_w = W[v]
            W[u] = durations[u] + max_w
            
    # Resource Intensity
    # Handle zero capacities
    safe_cap = np.where(capacities == 0, 1.0, capacities)
    # RI_i = sum(demands[i, r] / safe_cap[r])
    RI = np.sum(demands / safe_cap, axis=1)
    
    # Score Calculation
    # Avoid division by zero in TF+1 (TF >= 0)
    # Higher score = Higher Priority
    score = (W / (TF + 1.0)) * (1.0 + RI)
    
    return score
