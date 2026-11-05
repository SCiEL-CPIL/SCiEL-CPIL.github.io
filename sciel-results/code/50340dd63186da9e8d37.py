"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time


def priority(instance):
    """
    Dynamic resource-aware priority heuristic.
    
    Strategy:
    1. Compute earliest start times (EST) using forward pass (ignoring resources).
    2. For each activity, estimate the 'resource pressure' it would create at its EST.
       Resource pressure = sum over all resources of (demand / capacity) * (1 / (1 + idle_time_before))
       where idle_time_before is how much slack the resource has before this activity would start.
    3. Also incorporate bottom level (critical path distance to end) as a secondary factor.
    4. Combine: priority = alpha * bottom_level + beta * resource_pressure
    
    This differs from static heuristics because it considers how the activity's
    resource demands interact with the projected resource availability at its
    earliest start time, which is a dynamic aspect of the scheduling problem.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance.get("predecessors", [[] for _ in range(n)])
    
    # Get resource info
    # instance["demands"] is list of lists: demands[i][r] = demand of activity i on resource r
    # instance["capacities"] is list: capacities[r] = capacity of resource r
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    if demands is None or capacities is None:
        # Fallback to bottom level if no resource info
        return _bottom_level(n, durations, predecessors)
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    n_resources = len(capacities)
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Compute topological order
    in_degree = [0] * n
    for i in range(n):
        for pred in predecessors[i]:
            in_degree[i] += 1
    
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
        return durations.copy()
    
    # Forward pass: compute earliest start times (ignoring resources)
    est = np.zeros(n)
    for node in topo_order:
        if predecessors[node]:
            est[node] = max(est[p] + durations[p] for p in predecessors[node])
        else:
            est[node] = 0.0
    
    # Backward pass: compute bottom level (longest path to end)
    bottom_level = np.zeros(n)
    for node in reversed(topo_order):
        if not successors[node]:
            bottom_level[node] = durations[node]
        else:
            bottom_level[node] = durations[node] + max(bottom_level[s] for s in successors[node])
    
    # Compute resource pressure for each activity
    # For each activity i at its EST, how much pressure does it put on each resource?
    # We estimate the "tightness" by looking at what other activities would be running
    # at the same time (based on EST and durations) and how much capacity is left.
    
    # First, compute for each resource, the total demand that would be active
    # at each activity's EST based on the non-resource schedule
    resource_pressure = np.zeros(n)
    
    # For efficiency, precompute for each resource the list of (start, end, demand)
    # that would be active in the non-resource schedule
    for r in range(n_resources):
        if capacities[r] <= 0:
            continue
        
        # Collect all activities and their [EST, EST+dur) intervals with demand on resource r
        # For each activity i, compute how much of resource r is already used by others at EST[i]
        # This is O(n^2) per resource, but n=30 so it's fine
        
        for i in range(n):
            if demands[i][r] <= 0:
                continue
            
            start_i = est[i]
            end_i = est[i] + durations[i]
            
            # Sum demand from other activities j that overlap with [start_i, end_i)
            # and are not i itself
            total_other_demand = 0.0
            for j in range(n):
                if j == i:
                    continue
                if demands[j][r] <= 0:
                    continue
                start_j = est[j]
                end_j = est[j] + durations[j]
                # Check overlap between [start_i, end_i) and [start_j, end_j)
                overlap = min(end_i, end_j) - max(start_i, start_j)
                if overlap > 0:
                    total_other_demand += demands[j][r] * overlap / durations[i]  # normalized by duration of i
            
            # Pressure = how close we are to capacity
            # If total_other_demand + demands[i][r] > capacities[r], there's contention
            available = capacities[r] - total_other_demand
            if available <= 0:
                pressure = 1.0  # Maximum pressure
            else:
                pressure = min(1.0, demands[i][r] / available)
            
            resource_pressure[i] += pressure
    
    # Normalize resource_pressure to [0, 1]
    max_pressure = np.max(resource_pressure)
    if max_pressure > 0:
        resource_pressure = resource_pressure / max_pressure
    
    # Normalize bottom_level to [0, 1]
    max_bl = np.max(bottom_level)
    if max_bl > 0:
        bottom_level_norm = bottom_level / max_bl
    else:
        bottom_level_norm = bottom_level
    
    # Combine: weight bottom level heavily (it's the critical path driver)
    # and use resource pressure as a tie-breaker / secondary factor
    # Higher priority = schedule earlier
    alpha = 0.7
    beta = 0.3
    
    priority_scores = alpha * bottom_level_norm + beta * resource_pressure
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return priority_scores
    
    return priority_scores


def _bottom_level(n, durations, predecessors):
    """Fallback: compute bottom level (longest path to end)."""
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    in_degree = [0] * n
    for i in range(n):
        for pred in predecessors[i]:
            in_degree[i] += 1
    
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
        return durations.copy()
    
    bottom_level = np.zeros(n)
    for node in reversed(topo_order):
        if not successors[node]:
            bottom_level[node] = durations[node]
        else:
            bottom_level[node] = durations[node] + max(bottom_level[s] for s in successors[node])
    
    return bottom_level
# EVOLVE-BLOCK-END
