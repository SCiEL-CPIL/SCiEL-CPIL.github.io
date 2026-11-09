import numpy as np
import time

_START_TIME = None
_DEADLINE = 115.0  # safety margin before 120s kill


def priority(instance):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE - 5.0:
        # Fallback: just return durations-based priority
        durations = np.asarray(instance["durations"], dtype=float)
        return durations.copy()
    
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]
    
    n = len(durations)
    n_res = len(capacities)
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            
    # Compute in-degrees and topological order (Kahn's algorithm)
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1
            
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    temp_in_degree = in_degree.copy()
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            temp_in_degree[v] -= 1
            if temp_in_degree[v] == 0:
                queue.append(v)
                
    # Handle case where topo_order might not include all nodes (cycle or disconnected)
    # Add any missing nodes
    for i in range(n):
        if i not in topo_order:
            topo_order.append(i)
    
    # Calculate CPM Rank (longest path from i to end, including duration of i)
    cpm_rank = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if not successors[u]:
            cpm_rank[u] = durations[u]
        else:
            max_succ = max(cpm_rank[v] for v in successors[u])
            cpm_rank[u] = durations[u] + max_succ
            
    # Calculate resource load: sum of demand/capacity across all resources
    safe_caps = np.where(capacities == 0, 1.0, capacities)
    ratios = demands / safe_caps
    # Total resource load (sum across resources)
    resource_load = np.sum(ratios, axis=1)
    # Max resource pressure (single most constraining resource)
    resource_pressure = np.max(ratios, axis=1)
    
    # Successor count
    succ_count = np.array([len(s) for s in successors], dtype=float)
    
    # Normalize CPM rank to [0, 1]
    max_cpm = np.max(cpm_rank) if n > 0 else 1.0
    if max_cpm > 0:
        cpm_norm = cpm_rank / max_cpm
    else:
        cpm_norm = np.zeros(n, dtype=float)
    
    # Normalize resource load
    max_load = np.max(resource_load) if n > 0 else 1.0
    if max_load > 0:
        load_norm = resource_load / max_load
    else:
        load_norm = np.zeros(n, dtype=float)
    
    # Normalize resource pressure
    max_pressure = np.max(resource_pressure) if n > 0 else 1.0
    if max_pressure > 0:
        pressure_norm = resource_pressure / max_pressure
    else:
        pressure_norm = np.zeros(n, dtype=float)
    
    # Normalize successor count
    max_succ = np.max(succ_count) if n > 0 else 1.0
    if max_succ > 0:
        succ_norm = succ_count / max_succ
    else:
        succ_norm = np.zeros(n, dtype=float)
    
    # Combine scores with tuned weights
    # CPM rank is dominant: it tells us how much "work" remains on the critical path
    # Resource load: heavy activities should go early to free resources
    # Pressure: the most constraining single resource
    # Successor count: activities with many successors unlock more work
    
    w_cpm = 1.0
    w_load = 0.3
    w_pressure = 0.15
    w_succ = 0.1
    
    scores = (w_cpm * cpm_norm + 
              w_load * load_norm + 
              w_pressure * pressure_norm + 
              w_succ * succ_norm)
    
    # Add a small tie-breaker based on duration (shorter activities can finish sooner)
    # This helps when two activities have similar CPM ranks
    avg_dur = np.mean(durations) if n > 0 else 1.0
    if avg_dur > 0:
        dur_norm = durations / avg_dur
        scores = scores - 0.05 * dur_norm
    
    return scores
