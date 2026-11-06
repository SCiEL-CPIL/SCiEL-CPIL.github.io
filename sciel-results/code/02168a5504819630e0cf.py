"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque


def priority(instance):
    """
    Score each activity using a Min-Cut inspired Resource Bottleneck heuristic.
    
    This approach models the resource constraints as a network flow problem where
    the 'cut capacity' is determined by the tightest resource constraints.
    We approximate the min-cut impact by:
    1. Calculating global resource criticality (total demand / capacity).
    2. Weighting each activity's resource demand by the global criticality of that resource.
    3. Multiplying by the activity's critical path length (longest path to end).
    
    This prioritizes activities that are both on the critical path and consume
    resources that are heavily contested across the entire project, effectively
    identifying the 'bottleneck' activities that constrain the makespan.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Parse demands and capacities
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    
    # Handle case where demands might be 1D (single resource)
    if demands.ndim == 1:
        demands = demands.reshape(-1, 1)
        capacities = capacities.reshape(-1)
    
    num_resources = capacities.shape[0]
    
    # 1. Calculate Global Resource Criticality
    # total_demand[r] is the sum of demands for resource r across all activities
    total_demand = np.sum(demands, axis=0)
    safe_capacities = np.where(capacities > 0, capacities, 1.0)
    
    # criticality[r] = total_demand[r] / capacity[r]
    # This represents how 'tight' the resource is. Higher value = more bottleneck.
    resource_criticality = total_demand / safe_capacities
    
    # 2. Calculate Weighted Demand for each activity
    # weighted_demand[i] = sum(demands[i, r] * resource_criticality[r] for r)
    # This scales the activity's demand by how critical that resource is globally.
    weighted_demand = np.sum(demands * resource_criticality, axis=1)
    
    # 3. Compute Critical Path Length (Longest Path to End)
    predecessors = instance["predecessors"]
    
    # Build successor graph
    succ = defaultdict(list)
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        for p in preds:
            succ[p].append(i)
    
    # Topological order
    in_degree = [0] * n
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        in_degree[i] = len(preds)
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in succ[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    if len(topo_order) < n:
        topo_order = list(range(n))
    
    # Longest path calculation
    longest = np.zeros(n)
    for i in reversed(topo_order):
        max_succ = 0
        for s in succ[i]:
            if longest[s] > max_succ:
                max_succ = longest[s]
        longest[i] = durations[i] + max_succ
    
    # 4. Combine Scores
    # Priority is the product of the path length and the weighted resource demand.
    # This ensures that only activities on long paths that *also* use critical
    # resources get high priority.
    priority_scores = longest * weighted_demand
    
    # Add a small base priority to handle zero-demand or zero-length cases gracefully
    # and ensure unique ordering if possible
    priority_scores = priority_scores + 1e-6
    
    return priority_scores
# EVOLVE-BLOCK-END
