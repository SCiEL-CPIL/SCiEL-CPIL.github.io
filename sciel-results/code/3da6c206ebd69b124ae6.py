# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_priorities(instance):
    """
    Compute activity priorities based on a combination of:
    1. Longest duration (critical path proxy)
    2. Resource bottleneck centrality: sum over resources of (demand/capacity) 
       weighted by number of downstream activities
    3. Number of immediate successors (successor count)
    
    This aims to prioritize activities that are both long, consume scarce resources,
    and gate many downstream activities.
    """
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # shape (n, num_resources)
    capacities = np.asarray(instance["capacities"], dtype=float)  # shape (num_resources,)
    predecessors = instance["predecessors"]  # list of lists
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Compute number of downstream activities (including self) for each activity
    # Use topological order; assume DAG
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for pred in predecessors[i]:
            in_degree[i] += 1
    
    # Topological sort (Kahn's algorithm)
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    # If topo_order doesn't include all nodes, there's a cycle; fall back to simple ordering
    if len(topo_order) != n:
        topo_order = list(range(n))
    
    # Compute downstream count: number of activities that transitively depend on this one
    # Process in reverse topological order
    downstream_count = np.ones(n, dtype=float)
    for node in reversed(topo_order):
        for succ in successors[node]:
            downstream_count[node] += downstream_count[succ]
    
    # Compute resource bottleneck centrality
    # For each activity, compute sum over resources of (demand[r] / capacity[r]) * downstream_count
    # This measures how much scarce resource is consumed by this activity, weighted by its gating effect
    resource_pressure = np.zeros(n, dtype=float)
    for r in range(demands.shape[1]):
        if capacities[r] > 0:
            resource_pressure += (demands[:, r] / capacities[r]) * downstream_count
    
    # Combine features
    # Normalize each feature to [0, 1] range for fair combination
    def normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    norm_dur = normalize(durations)
    norm_pressure = normalize(resource_pressure)
    norm_downstream = normalize(downstream_count)
    
    # Weighted combination: emphasize duration and resource pressure
    weights = np.array([0.4, 0.35, 0.25])
    priority_scores = weights[0] * norm_dur + weights[1] * norm_pressure + weights[2] * norm_downstream
    
    # Add a small tie-breaker based on activity index to ensure deterministic but varied ordering
    priority_scores += 1e-6 * np.arange(n)
    
    return priority_scores


def priority(instance):
    """
    Returns a priority score for each activity. Higher score = higher priority.
    The serial schedule-generation scheme will start the highest-priority eligible activity.
    """
    start_time = time.time()
    
    try:
        scores = _compute_priorities(instance)
    except Exception:
        # Fallback to duration-based priority
        scores = np.asarray(instance["durations"], dtype=float)
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > 100:  # Safety margin before 120s limit
        return np.asarray(instance["durations"], dtype=float)
    
    return scores
# EVOLVE-BLOCK-END
