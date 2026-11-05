"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Compute a dynamic resource-contention priority for each activity.
    
    The priority combines:
    1. Longest path to the end (critical path length) - favors activities on the critical path
    2. Maximum resource contention ratio - favors activities that demand scarce resources
    3. A small tie-breaker based on duration to break ties deterministically
    
    This is a static computation (no simulation), but it captures resource scarcity
    by looking at the aggregate demand on each resource relative to capacity.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # shape (n_activities, n_resources)
    capacities = np.asarray(instance["capacities"], dtype=float)  # shape (n_resources,)
    predecessors = instance["predecessors"]  # list of lists, predecessors[i] = list of activity indices that must precede i
    
    n = len(durations)
    n_res = len(capacities)
    
    # Compute longest path to end (backward pass)
    # succ[i] = list of activities that depend on i
    succ = [[] for _ in range(n)]
    for i in range(n):
        for j in predecessors[i]:
            succ[j].append(i)
    
    # Latest start times (backward pass)
    # First, compute earliest finish times (forward pass)
    earliest_start = np.zeros(n, dtype=float)
    for i in range(n):
        if predecessors[i]:
            earliest_start[i] = max(earliest_start[p] + durations[p] for p in predecessors[i])
    
    # Project duration
    project_duration = max(earliest_start[i] + durations[i] for i in range(n))
    
    # Latest start times
    latest_start = np.zeros(n, dtype=float)
    # Process in reverse topological order
    # Use a simple approach: iterate until convergence or use reverse topological sort
    # For simplicity, do a reverse pass assuming topological order is given by index (common in these problems)
    # But to be safe, let's do a proper backward pass
    
    # Actually, let's just compute the "distance to end" which is the longest path from i to any sink
    dist_to_end = np.zeros(n, dtype=float)
    # Process nodes in reverse topological order
    # We need topological order. Let's compute it.
    in_degree = np.array([len(preds) for preds in predecessors], dtype=int)
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_degree_copy = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succ[u]:
            in_degree_copy[v] -= 1
            if in_degree_copy[v] == 0:
                queue.append(v)
    
    # If topo_order doesn't include all nodes (cycle), fall back to index order
    if len(topo_order) != n:
        topo_order = list(range(n))
    
    # Backward pass: dist_to_end[i] = durations[i] + max(dist_to_end[j] for j in succ[i])
    # Process in reverse topological order
    dist_to_end = np.zeros(n, dtype=float)
    for i in reversed(topo_order):
        if succ[i]:
            dist_to_end[i] = durations[i] + max(dist_to_end[j] for j in succ[i])
        else:
            dist_to_end[i] = durations[i]
    
    # Compute resource contention: for each resource, total demand of all activities that use it
    # resource_load[r] = sum of demands[i][r] for all activities i
    resource_load = np.sum(demands, axis=0)  # shape (n_resources,)
    
    # Resource scarcity ratio: load / capacity (higher means more contended)
    # Avoid division by zero
    safe_capacities = np.where(capacities > 0, capacities, 1.0)
    resource_scarcity = resource_load / safe_capacities  # shape (n_resources,)
    
    # For each activity, compute its max contention: max over resources r of (demands[i][r] * resource_scarcity[r] / max(demands[i][r], 1))
    # Actually, a simpler measure: for each activity, the max over resources of (demand[i][r] / capacity[r])
    # This captures how much of the resource this single activity would consume
    # But we also want to account for total contention. Let's use:
    # contention[i] = max over r of (demands[i][r] * resource_scarcity[r])
    # This is high if the activity demands a lot of a scarce resource
    
    # demands shape (n, n_res), resource_scarcity shape (n_res,)
    # contention[i] = max_r (demands[i][r] * resource_scarcity[r])
    if n_res > 0:
        weighted_demand = demands * resource_scarcity[np.newaxis, :]  # shape (n, n_res)
        contention = np.max(weighted_demand, axis=1)  # shape (n,)
    else:
        contention = np.zeros(n, dtype=float)
    
    # Normalize both components to [0, 1] for combination
    dist_range = dist_to_end.max() - dist_to_end.min() if n > 0 else 1.0
    if dist_range > 0:
        dist_norm = (dist_to_end - dist_to_end.min()) / dist_range
    else:
        dist_norm = np.ones(n, dtype=float)
    
    cont_range = contention.max() - contention.min() if n > 0 else 1.0
    if cont_range > 0:
        cont_norm = (contention - contention.min()) / cont_range
    else:
        cont_norm = np.zeros(n, dtype=float)
    
    # Combine: weight critical path more heavily, but also prioritize scarce-resource activities
    # priority = alpha * dist_norm + (1 - alpha) * cont_norm
    # Let's use alpha = 0.7 to favor critical path, with resource contention as secondary
    alpha = 0.7
    scores = alpha * dist_norm + (1.0 - alpha) * cont_norm
    
    # Add a tiny tie-breaker based on duration to ensure deterministic ordering
    dur_range = durations.max() - durations.min() if n > 0 else 1.0
    if dur_range > 0:
        dur_norm = (durations - durations.min()) / dur_range
    else:
        dur_norm = np.zeros(n, dtype=float)
    scores = scores + 0.01 * dur_norm
    
    return scores
# EVOLVE-BLOCK-END
