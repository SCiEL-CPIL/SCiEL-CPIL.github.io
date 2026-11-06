"""Rewritten program for Resource-Constrained Project Scheduling."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    
    Uses a Total Float (Slack) based heuristic combined with Resource Pressure.
    
    Approach:
    1. Compute Total Float for each activity using Forward and Backward passes.
       Activities with lower Total Float (closer to critical path) are prioritized.
    2. Incorporate Resource Demand: Activities with higher resource consumption
       are prioritized to resolve resource contention early.
    3. Incorporate Duration: Shorter tasks are slightly preferred to allow
       more flexibility in scheduling successors.
    
    This differs from the "Longest Path" approach by considering the global
    position of the task (earliest start constraints) rather than just the
    remaining path length.
    """
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance["predecessors"]
    
    # Build successor list and in-degrees for topological sort
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        for i in predecessors[j]:
            if i is not None:
                successors[i].append(j)
                in_degree[j] += 1
    
    # Topological Sort (Kahn's Algorithm)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
                
    # Handle potential cycles or disconnected nodes (should not happen in valid RCPSP)
    if len(topo_order) < n:
        remaining = [i for i in range(n) if i not in set(topo_order)]
        topo_order.extend(remaining)

    # Forward Pass: Calculate Earliest Start (ES) and Earliest Finish (EF)
    ES = np.zeros(n, dtype=float)
    EF = np.zeros(n, dtype=float)
    for node in topo_order:
        if predecessors[node]:
            # Max EF of all predecessors
            pred_ef_max = max(EF[p] for p in predecessors[node] if p is not None) if predecessors[node] else 0.0
            ES[node] = pred_ef_max
        else:
            ES[node] = 0.0
        EF[node] = ES[node] + durations[node]
        
    # Project Makespan (Earliest Finish of the project)
    makespan = max(EF) if n > 0 else 0.0

    # Backward Pass: Calculate Latest Finish (LF) and Latest Start (LS)
    LF = np.zeros(n, dtype=float)
    LS = np.zeros(n, dtype=float)
    
    # Initialize LF for nodes with no successors to the project makespan
    for node in reversed(topo_order):
        if not successors[node]:
            LF[node] = makespan
        else:
            # Min LS of all successors
            succ_ls_min = min(LS[s] for s in successors[node])
            LF[node] = succ_ls_min
        LS[node] = LF[node] - durations[node]
        
    # Calculate Total Float (Slack)
    # TF = LS - ES
    total_float = LS - ES
    
    # Resource Demand Pressure
    demands = instance["demands"]
    # Sum demands across resources to get total resource intensity
    total_demand = np.sum(np.asarray(demands, dtype=float), axis=1)
    
    # Normalization
    # We want to prioritize LOW Total Float. So we invert it.
    max_float = np.max(total_float) if np.max(total_float) > 0 else 1.0
    # Invert float: 0 float -> 1.0, max float -> 0.0
    norm_float_inv = 1.0 - (total_float / max_float)
    
    max_demand = np.max(total_demand) if np.max(total_demand) > 0 else 1.0
    norm_demand = total_demand / max_demand
    
    max_dur = np.max(durations) if np.max(durations) > 0 else 1.0
    norm_dur = durations / max_dur
    
    # Combine scores
    # Primary: Low Total Float (High Criticality)
    # Secondary: High Resource Demand
    # Tertiary: Short Duration (lower duration is better, so 1 - norm_dur)
    
    # Weights: 0.7 for Criticality (Float), 0.2 for Demand, 0.1 for Duration
    score = 0.7 * norm_float_inv + 0.2 * norm_demand + 0.1 * (1.0 - norm_dur)
    
    return score
# EVOLVE-BLOCK-END
