"""
Combined program for Resource-Constrained Project Scheduling.

Strategy:
- Uses the multiplicative "Resource-Weighted Critical Path" approach from Inspiration 1, 
  which outperformed the additive approach in the current program.
- The priority score is defined as: BL * (1 + NormalizedResourcePressure).
- BL (Bottom Level) is the longest path from the activity to the project end.
- ResourcePressure is calculated by summing (demand/capacity) ratios, where each ratio 
  is weighted by the total project load on that resource. This amplifies the priority 
  of activities using globally scarce resources.
- Robust data parsing and topological sorting are retained from the current program.
"""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time

def priority(instance):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin

    # 1. Parse Instance Data
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    predecessors = instance.get("predecessors", None)
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
        
    # Normalize predecessors to lists of ints
    pred_lists = []
    for i in range(n):
        p = predecessors[i]
        if p is None:
            p = []
        if isinstance(p, (list, tuple, np.ndarray)):
            p = [int(x) for x in p]
        else:
            p = [int(p)]
        # Filter out self and out-of-range
        p = [x for x in p if 0 <= x < n and x != i]
        pred_lists.append(p)

    # Build successor lists
    succ_lists = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in pred_lists[i]:
            succ_lists[p].append(i)
            in_degree[i] += 1

    # 2. Topological Sort (Kahn's Algorithm)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    
    # Use a copy of in_degree for the sort
    temp_in_degree = in_degree.copy()
    
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in succ_lists[node]:
            temp_in_degree[succ] -= 1
            if temp_in_degree[succ] == 0:
                queue.append(succ)
                
    # Handle potential cycles (fallback to original order if cycle detected, though RCPSP should be acyclic)
    if len(topo_order) < n:
        remaining = set(range(n)) - set(topo_order)
        topo_order.extend(list(remaining))

    # 3. Calculate Bottom Level (BL)
    # BL[i] = durations[i] + max(BL[s] for s in successors[i])
    BL = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        if succ_lists[node]:
            max_succ_bl = max(BL[s] for s in succ_lists[node])
        else:
            max_succ_bl = 0.0
        BL[node] = durations[node] + max_succ_bl

    # 4. Calculate Resource Pressure
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    resource_pressure = np.zeros(n, dtype=float)
    
    if demands is not None and capacities is not None:
        demands_arr = np.asarray(demands, dtype=float)
        capacities_arr = np.asarray(capacities, dtype=float)
        
        # Handle shape variations
        if demands_arr.ndim == 1:
            demands_arr = demands_arr.reshape(n, 1)
        if demands_arr.shape[0] != n:
            # Fallback if shape mismatch
            demands_arr = np.zeros((n, 1), dtype=float)
            if demands_arr.size > 0:
                demands_arr[0, 0] = 1.0 # Placeholder to avoid empty array issues in logic below, though we re-init pressure to 0
        
        R = demands_arr.shape[1]
        
        if capacities_arr.size == 1:
            caps = np.full(R, float(capacities_arr[0]), dtype=float)
        else:
            caps = capacities_arr.reshape(-1)
            if caps.size < R:
                tmp = np.ones(R, dtype=float)
                tmp[:caps.size] = caps
                caps = tmp
            else:
                caps = caps[:R]
        
        # Ensure no division by zero
        caps = np.maximum(caps, 1e-9)
        
        # Per-activity resource usage ratio
        ratio = demands_arr / caps[np.newaxis, :]  # (n, R)
        
        # Weight by resource scarcity (how much total demand is on this resource)
        # This makes resources with high total load more "expensive"
        resource_load = np.sum(ratio, axis=0)  # (R,)
        
        load_max = float(np.max(resource_load)) if R > 0 else 1.0
        if load_max > 1e-12:
            load_weight = resource_load / load_max  # (R,)
        else:
            load_weight = np.ones(R, dtype=float)
            
        # Activity pressure: weighted sum of ratios
        raw_pressure = np.sum(ratio * load_weight[np.newaxis, :], axis=1)  # (n,)
        
        # Normalize pressure to [0, 1]
        rp_max = float(np.max(raw_pressure)) if n > 0 else 1.0
        if rp_max > 1e-12:
            resource_pressure = raw_pressure / rp_max
        else:
            resource_pressure = np.zeros(n, dtype=float)
            
    # 5. Combine Scores
    # Multiplicative interaction: High BL * High Pressure = Very High Priority
    # This captures that critical activities using scarce resources are most urgent.
    score = BL * (1.0 + resource_pressure)
    
    # Tie-breaker: Slightly favor longer durations to break ties in BL/Pressure
    dur_max = float(np.max(durations)) if n > 0 else 1.0
    if dur_max > 1e-12:
        dur_norm = durations / dur_max
        score += 0.01 * dur_norm
    else:
        score += 0.0

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return score

    return np.asarray(score, dtype=float)
# EVOLVE-BLOCK-END
