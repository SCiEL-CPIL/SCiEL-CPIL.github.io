import numpy as np
import time

def priority(instance):
    """
    Scores activities for a Serial Schedule Generation Scheme.
    Uses a composite heuristic based on Duration, Total Float (from CPM),
    and Resource Intensity.
    """
    start_time = time.time()
    deadline = 150.0 # Safety margin from 120s limit? No, 120s is the limit. 150s is the kill switch.
    # The function itself should be fast. CPM on 30 nodes is O(N^2) or O(N+E), very fast.
    
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance["predecessors"]
    demands = instance["demands"]
    capacities = np.asarray(instance["capacities"], dtype=float)
    
    n = len(durations)
    
    # 1. Build Successors list for backward pass
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            
    # 2. Forward Pass (ES, EF)
    ES = np.zeros(n)
    EF = np.zeros(n)
    
    # Topological sort is implied by the predecessor structure usually, 
    # but simple iteration works if predecessors are always lower index? 
    # Not guaranteed. We need a proper topological order or iterative relaxation.
    # Given N=30, we can just do a simple DFS or Kahn's algorithm to get topo order.
    
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1
            
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        # Pick first node (order doesn't matter for correctness of CPM, just for processing)
        u = queue.pop(0) 
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # If topo_order length < n, there's a cycle. Assume valid instance.
    for u in topo_order:
        ES[u] = max([EF[p] for p in predecessors[u]], default=0.0)
        EF[u] = ES[u] + durations[u]
        
    project_duration = max(EF) if n > 0 else 0.0
    
    # 3. Backward Pass (LS, LF)
    LS = np.zeros(n)
    LF = np.zeros(n)
    
    # Reverse topological order
    for u in reversed(topo_order):
        if not successors[u]:
            LF[u] = project_duration
        else:
            LF[u] = min([LS[s] for s in successors[u]])
        LS[u] = LF[u] - durations[u]
        
    # 4. Total Float
    TF = LS - ES
    
    # 5. Resource Intensity
    # demands is likely a list of lists: demands[i][r]
    # capacities is a list: capacities[r]
    n_res = len(capacities)
    Load = np.zeros(n)
    for i in range(n):
        load_val = 0.0
        for r in range(n_res):
            if capacities[r] > 0:
                load_val += demands[i][r] / capacities[r]
            else:
                # If capacity is 0 and demand is > 0, it's infeasible. 
                # If demand is 0, it's 0.
                if demands[i][r] > 0:
                    load_val += 1000.0 # High penalty for infeasible/scarce
                else:
                    load_val += 0.0
        Load[i] = load_val
        
    # 6. Composite Score
    # We want high priority for:
    # - High Duration
    # - Low Total Float (Critical)
    # - High Resource Load
    
    # Normalize to avoid scale issues?
    max_dur = np.max(durations) if n > 0 else 1.0
    max_tf = np.max(TF) if n > 0 else 1.0
    max_load = np.max(Load) if n > 0 else 1.0
    
    # Avoid division by zero in normalization
    if max_dur == 0: max_dur = 1.0
    if max_tf == 0: max_tf = 1.0
    if max_load == 0: max_load = 1.0
    
    # Score = Weighted Sum
    # Criticality: (1 - TF/max_tf) -> 1 for critical, 0 for max slack
    # Duration: Dur/max_dur
    # Load: Load/max_load
    
    Criticality = 1.0 - (TF / max_tf)
    NormDur = durations / max_dur
    NormLoad = Load / max_load
    
    # Weights: Criticality is usually most important.
    score = 0.5 * Criticality + 0.3 * NormDur + 0.2 * NormLoad
    
    return score
