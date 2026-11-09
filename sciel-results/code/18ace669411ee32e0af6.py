import numpy as np
from collections import deque

def priority(instance):
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    preds = instance["predecessors"]
    
    # Build adjacency lists
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    # Topological sort
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
            
    topo_order = []
    in_degree_copy = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree_copy[v] -= 1
            if in_degree_copy[v] == 0:
                queue.append(v)
                
    if len(topo_order) != n:
        # Fallback
        return -durations # Lower duration = higher priority? Or just durations?
        # Let's return -TF logic equivalent. If invalid, just return -durations as a heuristic.
        
    # Forward Pass: Earliest Start (ES), Earliest Finish (EF)
    ES = np.zeros(n, dtype=float)
    EF = np.zeros(n, dtype=float)
    for u in topo_order:
        if preds[u]:
            ES[u] = max(EF[p] for p in preds[u])
        else:
            ES[u] = 0.0
        EF[u] = ES[u] + durations[u]
        
    # Unconstrained Project Duration
    C_max = np.max(EF) if n > 0 else 0.0
    
    # Backward Pass: Latest Finish (LF), Latest Start (LS)
    LF = np.full(n, C_max, dtype=float)
    LS = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if successors[u]:
            LF[u] = min(LS[v] for v in successors[u])
        else:
            LF[u] = C_max
        LS[u] = LF[u] - durations[u]
        
    # Total Float
    TF = LS - ES
    
    # Resource Demand Intensity
    # Max demand ratio per activity
    demand_ratio = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(len(capacities)):
            if capacities[j] > 0:
                ratio = demands[i][j] / capacities[j]
                if ratio > demand_ratio[i]:
                    demand_ratio[i] = ratio
    
    # Normalize TF and Demand Ratio for combination
    # We want Low TF to be High Priority.
    # Score = -TF is good.
    # To break ties, we prefer Lower Demand Ratio (lighter tasks) to free resources sooner.
    # So subtract a small multiple of Demand Ratio.
    
    # Normalize to [0, 1] to control weight
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
        
    tf_norm = normalize(TF) # 0 is low TF (critical), 1 is high TF (slack)
    demand_norm = normalize(demand_ratio) # 0 is low demand, 1 is high demand
    
    # We want high score for low TF.
    # So base score is (1 - tf_norm).
    # We want high score for low demand (tie breaker).
    # So subtract a small amount of demand_norm.
    
    # Weight of demand penalty should be small, e.g., 0.1
    scores = (1.0 - tf_norm) - 0.1 * demand_norm
    
    return scores
