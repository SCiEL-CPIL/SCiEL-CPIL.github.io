"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def priority(instance):
    """
    Compute priority scores using Minimum Total Float (MTF) heuristic.
    
    Strategy:
    1. Compute Early Start (ES) and Late Start (LS) times ignoring resource constraints.
    2. Calculate Total Float (TF = LS - ES) for each activity.
    3. Prioritize activities with the LOWEST Total Float (highest criticality).
    4. Use Resource Demand as a tie-breaker for activities with similar float.
    
    This shifts from a 'Weighted Path Length' model to a 'Slack/Flexibility' model,
    which is a standard and robust heuristic for RCPSP that often outperforms simple CP length
    by focusing on scheduling flexibility rather than just structural distance.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance.get("predecessors", [[] for _ in range(n)])
    
    if "demands" not in instance or "capacities" not in instance:
        return _static_float_priority(instance)
        
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    
    # 1. Compute ES and LS (Forward and Backward Pass)
    es, ls = _compute_es_ls(n, durations, predecessors)
    
    if es is None or ls is None:
        # Fallback to durations if cycle or error
        return durations.copy()
        
    # 2. Total Float
    # TF = LS - ES
    total_float = ls - es
    
    # 3. Priority Score
    # We want high priority for LOW float.
    # Score = -Total Float.
    # To handle ties and add resource awareness:
    # If two activities have the same float, prioritize the one with higher resource demand.
    
    # Normalize float to a small range to use as tie-breaker?
    # Or just use -TF as primary, and Demand as secondary?
    # The interface returns a single score.
    # We can combine them: Score = -TF * (1 + alpha * Demand_Norm)
    # Or: Score = -TF + beta * Demand_Norm
    
    # Let's try: Score = -Total Float
    # But pure MTF can be unstable if many activities have 0 float.
    # Let's add a small resource weight to break ties.
    
    # Calculate max demand for normalization
    max_demand = np.max(np.sum(demands, axis=1)) if demands.size > 0 else 1.0
    if max_demand == 0:
        max_demand = 1.0
        
    resource_load = np.sum(demands, axis=1) / max_demand
    
    # Combine: Primary is -TF. Secondary is Resource Load.
    # If TF is 0 for many, Resource Load decides.
    # If TF is 5 and 6, the difference is 1. Resource load is 0..1.
    # We need to ensure TF dominates.
    # Scale TF down? No, TF is in time units.
    # Let's use: Priority = -Total Float + 0.1 * Resource Load
    # This ensures that a difference of 1 time unit in float outweighs the resource tie-breaker.
    
    priority_scores = -total_float + 0.1 * resource_load
    
    # Check time
    if time.time() > deadline:
        return -total_float
        
    return priority_scores

def _compute_es_ls(n, durations, predecessors):
    """
    Compute Early Start (ES) and Late Start (LS) times for all activities.
    Returns (es_array, ls_array) or (None, None) on failure.
    """
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
            in_degree[i] += 1
            
    # Topological Sort
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
                
    if len(topo_order) != n:
        return None, None
        
    # Forward Pass: Compute ES
    es = np.zeros(n)
    for node in topo_order:
        if predecessors[node]:
            es[node] = max(es[p] + durations[p] for p in predecessors[node])
        else:
            es[node] = 0.0
            
    # Project Duration (Max EF)
    project_duration = max(es[i] + durations[i] for i in range(n))
    
    # Backward Pass: Compute LS
    ls = np.zeros(n)
    # Initialize LS for sinks (no successors)
    for node in reversed(topo_order):
        if not successors[node]:
            ls[node] = project_duration - durations[node]
        else:
            # LS = min(LS of successors) - Duration
            ls[node] = min(ls[s] for s in successors[node]) - durations[node]
            
    return es, ls

def _static_float_priority(instance):
    """Fallback to static float if resources are not provided."""
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance.get("predecessors", [[] for _ in range(n)])
    es, ls = _compute_es_ls(n, durations, predecessors)
    if es is None:
        return durations.copy()
    total_float = ls - es
    return -total_float
# EVOLVE-BLOCK-END
