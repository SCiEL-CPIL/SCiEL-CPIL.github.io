import numpy as np

def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]
    
    n = len(durations)
    
    # 1. Build successors list and in-degrees
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    # 2. Topological Sort (Kahn's Algorithm)
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # Handle potential cycles (should not happen in RCPSP)
    if len(topo_order) != n:
        safe_capacities = np.where(capacities == 0, 1.0, capacities)
        pressure = np.sum(demands / safe_capacities, axis=1)
        return pressure

    # 3. Compute Earliest Start (ES)
    ES = [0.0] * n
    for u in topo_order:
        for v in successors[u]:
            if ES[u] + durations[u] > ES[v]:
                ES[v] = ES[u] + durations[u]
                
    # 4. Compute Makespan (T)
    T = 0.0
    for i in range(n):
        if ES[i] + durations[i] > T:
            T = ES[i] + durations[i]
            
    # 5. Compute Longest Path to End (CPM) in reverse topological order
    path_to_end = [0.0] * n
    for u in reversed(topo_order):
        max_succ = 0.0
        for v in successors[u]:
            if path_to_end[v] > max_succ:
                max_succ = path_to_end[v]
        path_to_end[u] = durations[u] + max_succ
        
    # 6. Compute Resource Pressure
    safe_capacities = np.where(capacities == 0, 1.0, capacities)
    pressure = np.sum(demands / safe_capacities, axis=1)
    
    # 7. Combine Scores: Critical Path - Slack + Pressure
    # Slack = (T - path_to_end[i]) - ES[i]
    scores = np.zeros(n)
    for i in range(n):
        slack = (T - path_to_end[i]) - ES[i]
        if slack < 0:
            slack = 0.0
        scores[i] = path_to_end[i] - slack + pressure[i]
        
    return scores
