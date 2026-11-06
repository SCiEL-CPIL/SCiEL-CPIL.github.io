"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def _get_predecessors(instance, N):
    """Extract and normalize predecessor lists."""
    preds = instance.get("predecessors", None)
    if preds is None:
        for key in ["preds", "predecessor", "predecessors_list"]:
            if key in instance:
                preds = instance[key]
                break
        if preds is None:
            preds = [[] for _ in range(N)]
    else:
        if isinstance(preds, np.ndarray):
            preds = preds.tolist()
        # Ensure it's a list of lists of ints
        normalized_preds = []
        for p in preds:
            if isinstance(p, (list, tuple, np.ndarray)):
                normalized_preds.append([int(x) for x in p])
            else:
                # If it's a single int, treat as list of one
                normalized_preds.append([int(p)])
        preds = normalized_preds
    return preds

def _compute_priorities(instance):
    N = len(instance["durations"])
    if N == 0:
        return np.array([], dtype=float)
        
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # (N, R)
    capacities = np.asarray(instance["capacities"], dtype=float)  # (R,)
    
    R = len(capacities)
    preds = _get_predecessors(instance, N)
    
    # Build successor lists
    succs = [[] for _ in range(N)]
    for j in range(N):
        for i in preds[j]:
            if 0 <= i < N:
                succs[i].append(j)
                
    # Topological sort
    in_degree = np.zeros(N, dtype=int)
    for j in range(N):
        for i in preds[j]:
            if 0 <= i < N:
                in_degree[j] += 1
                
    queue = deque([i for i in range(N) if in_degree[i] == 0])
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in succs[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # If topological sort didn't include all nodes (cycle?), fallback
    if len(topo) < N:
        return np.asarray(durations, dtype=float)
        
    # 1. Bottom Level (BL): Longest path from node to end (sink)
    # BL[i] = duration[i] + max(BL[j] for j in succs[i])
    # We process in reverse topological order
    BL = np.zeros(N, dtype=float)
    for u in reversed(topo):
        if succs[u]:
            BL[u] = durations[u] + max(BL[v] for v in succs[u])
        else:
            BL[u] = durations[u]
            
    # 2. Total Successors (TS): Count of all direct and indirect successors
    TS = np.zeros(N, dtype=int)
    for u in reversed(topo):
        if succs[u]:
            TS[u] = len(succs[u]) + sum(TS[v] for v in succs[u])
        else:
            TS[u] = 0
            
    # 3. Resource Pressure
    # Max ratio of demand to capacity
    if R > 0 and np.all(capacities > 0):
        # Avoid division by zero
        safe_cap = np.where(capacities > 0, capacities, 1.0)
        pressure_matrix = demands / safe_cap  # (N, R)
        # Use max pressure per activity
        res_pressure = np.max(pressure_matrix, axis=1)
    else:
        res_pressure = np.zeros(N, dtype=float)
        
    # Normalization
    def normalize(x):
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            return x
        min_val = np.min(x)
        max_val = np.max(x)
        rng = max_val - min_val
        if rng < 1e-12:
            return np.zeros_like(x)
        return (x - min_val) / rng
        
    norm_BL = normalize(BL)
    norm_TS = normalize(TS)
    norm_pressure = normalize(res_pressure)
    
    # Combination Strategy
    # Based on insight: "Bottom-Level ... multiplied by resource scarcity"
    # Let's try: Priority = norm_BL * (1 + norm_pressure) + 0.2 * norm_TS
    # This boosts BL if the activity is resource-heavy.
    
    # Alternative: Weighted Sum
    # w_BL = 0.6, w_Pressure = 0.3, w_TS = 0.1
    # Let's stick to the multiplicative boost for BL as it was noted as improving in another island.
    
    priority = norm_BL * (1.0 + norm_pressure) + 0.2 * norm_TS
    
    return priority

def priority(instance):
    """
    Returns a priority score for each activity.
    Higher priority = scheduled sooner in serial SGS.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    try:
        N = len(instance["durations"])
        if N == 0:
            return np.array([], dtype=float)
            
        prio = _compute_priorities(instance)
        prio = np.asarray(prio, dtype=float)
        
        if len(prio) != N:
            raise ValueError("Priority length mismatch")
            
        return prio
        
    except Exception:
        # Fallback: Duration-based
        try:
            return np.asarray(instance["durations"], dtype=float)
        except Exception:
            N = len(instance.get("durations", []))
            return np.zeros(N, dtype=float)
# EVOLVE-BLOCK-END
