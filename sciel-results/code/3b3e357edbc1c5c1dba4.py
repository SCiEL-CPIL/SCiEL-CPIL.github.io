import numpy as np
import time

def priority(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    cycle_time = instance.get("cycle_time", 40.0)
    
    # Build successor list
    succ = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in preds[i]:
            succ[p].append(i)
            in_degree[i] += 1
    
    # Compute bottom level (longest path to any sink) - total downstream processing time
    bottom_level = times.copy()
    od = [len(succ[i]) for i in range(n)]
    od_work = od[:]
    q = [i for i in range(n) if od_work[i] == 0]
    
    while q:
        u = q.pop()
        for p in preds[u]:
            new_bl = times[p] + bottom_level[u]
            if new_bl > bottom_level[p]:
                bottom_level[p] = new_bl
            od_work[p] -= 1
            if od_work[p] == 0:
                q.append(p)
    
    # Compute minimum number of stations required for downstream subtree
    # This is the ceiling of (bottom_level[i]) / cycle_time, but we need to be more precise
    # Use a DP: min_stations[i] = max(1, ceil((times[i] + sum of parallel branches) / cycle_time))
    # Approximation: use ceil(bottom_level[i] / cycle_time) as proxy for station scarcity
    
    # Compute slack: how much cycle time is left after this task's critical path
    # slack[i] = cycle_time - bottom_level[i] (if positive, there's room; if negative, tight)
    slack = cycle_time - bottom_level
    
    # Bottleneck density: time / (minimum stations needed for downstream)
    # Higher = more critical, should be scheduled earlier
    
    # Minimum stations needed: at least 1, and at least ceil(bottom_level / cycle_time)
    min_stations = np.maximum(1.0, np.ceil(bottom_level / max(cycle_time, 1e-10)))
    
    # Bottleneck density: processing time relative to station scarcity
    # Tasks with high time and high min_stations are bottlenecks
    bottleneck_density = times / (min_stations + 1e-10)
    
    # Also compute RPV for additional signal
    rpv = times.copy()
    od2 = od[:]
    q2 = [i for i in range(n) if od2[i] == 0]
    while q2:
        u = q2.pop()
        for p in preds[u]:
            rpv[p] += rpv[u]
            od2[p] -= 1
            if od2[p] == 0:
                q2.append(p)
    
    # Normalize components
    def normalize(arr):
        arr = np.asarray(arr, dtype=float)
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val < 1e-10:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    bd_norm = normalize(bottleneck_density)
    bl_norm = normalize(bottom_level)
    rpv_norm = normalize(rpv)
    
    # Combine: bottleneck density is primary, bottom level secondary, RPV tertiary
    # Weight bottleneck density heavily since it directly captures the "tight slot" concern
    combined = 0.6 * bd_norm + 0.25 * bl_norm + 0.15 * rpv_norm
    
    # Check deadline
    if time.time() > deadline:
        return combined
    
    return combined
