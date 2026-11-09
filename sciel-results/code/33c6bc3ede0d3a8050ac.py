"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE_SECONDS = 110.0  # Safety margin before 120s kill


def _compute_priority(instance):
    """
    Compute a priority score for each activity to minimize makespan.
    
    Strategy: Hybrid of Critical Path (Head/Tail/Slack) and Resource Pressure.
    1. Compute Head, Tail, and Slack (Total Float).
    2. Compute Resource Demand Intensity (sum of demand/capacity).
    3. Use a smooth surrogate (log-sum-exp based soft-max/min) to blend
       the urgency signals (Low Slack, High Tail) with resource contention.
    4. The score is normalized such that higher is better.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # shape: (n_activities, n_resources)
    capacities = np.asarray(instance["capacities"], dtype=float)  # shape: (n_resources,)
    predecessors = instance["predecessors"]  # list of lists or similar
    
    n = len(durations)
    if n == 0:
        return np.array([])
    
    # Convert predecessors to a consistent format
    if isinstance(predecessors, np.ndarray):
        pred_lists = []
        for i in range(n):
            pred_lists.append(list(predecessors[i]))
    else:
        pred_lists = [list(p) for p in predecessors]
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in pred_lists[i]:
            if i not in successors[p]:
                successors[p].append(i)
    
    # Topological order
    in_degree = [len(pred_lists[i]) for i in range(n)]
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If topo_order doesn't contain all nodes, there's a cycle. 
    # Fallback to simple duration priority if cycle detected.
    if len(topo_order) < n:
        return durations.copy()

    # head[i] = longest duration from any source to i (including i's own duration)
    head = np.zeros(n, dtype=float)
    for i in topo_order:
        if pred_lists[i]:
            head[i] = durations[i] + max(head[p] for p in pred_lists[i])
        else:
            head[i] = durations[i]
    
    # tail[i] = longest duration from i to any sink (including i's own duration)
    tail = np.zeros(n, dtype=float)
    for i in reversed(topo_order):
        if not successors[i]:
            tail[i] = durations[i]
        else:
            tail[i] = durations[i] + max(tail[j] for j in successors[i])
    
    # Project makespan without resource constraints (longest path)
    makespan_ignoring_resources = max(tail) if n > 0 else 0.0
    
    # Total Float / Slack
    # slack[i] = makespan - head[i] - tail[i] + durations[i]
    slack = np.zeros(n, dtype=float)
    for i in range(n):
        slack[i] = makespan_ignoring_resources - head[i] - tail[i] + durations[i]
        
    # Resource demand intensity
    # For each activity, sum the ratio of demand to capacity for each resource.
    # This highlights activities that consume a large fraction of available resources.
    demand_intensity = np.zeros(n, dtype=float)
    n_resources = len(capacities)
    for i in range(n):
        for r in range(n_resources):
            cap = capacities[r]
            if cap > 0:
                demand_intensity[i] += demands[i][r] / cap
            else:
                # If capacity is 0 and demand is > 0, this is infeasible, but 
                # we assign a high penalty to prioritize it? No, it's infeasible.
                # Assume valid instances.
                demand_intensity[i] += 0.0

    # Normalization helper
    def normalize(arr):
        arr = np.asarray(arr, dtype=float)
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-10:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    # We want to maximize:
    # 1. Tail (longest path to end) -> High is good
    # 2. -Slack (urgency) -> High is good (Low slack is good)
    # 3. Demand Intensity (resource pressure) -> High is good (scarce resource usage)
    
    # Smooth combination using a weighted sum of normalized components.
    # The current program used lexicographic ranking which is brittle.
    # Inspiration 1 used 0.6*Tail + 0.4*(-Slack).
    # Inspiration 2 added resource contention.
    # We will use a smooth blend:
    # Score = w1 * Norm(Tail) + w2 * Norm(-Slack) + w3 * Norm(Demand)
    
    # Tuning weights:
    # Critical path structure is usually the dominant factor.
    # Resource contention is a secondary factor that breaks ties or adjusts for bottlenecks.
    w_tail = 0.5
    w_slack = 0.3
    w_demand = 0.2
    
    norm_tail = normalize(tail)
    norm_slack_inv = normalize(-slack)
    norm_demand = normalize(demand_intensity)
    
    priority_scores = (w_tail * norm_tail + 
                       w_slack * norm_slack_inv + 
                       w_demand * norm_demand)
    
    return priority_scores


def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority (scheduled first when eligible).
    """
    start_time = time.time()
    
    try:
        scores = _compute_priority(instance)
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE_SECONDS:
            # Should not happen for 30 activities, but guard anyway
            return np.asarray(instance["durations"], dtype=float)
        return scores
    except Exception:
        # Fallback: longest processing time
        try:
            return np.asarray(instance["durations"], dtype=float)
        except Exception:
            n = len(instance.get("durations", []))
            return np.ones(n, dtype=float)
# EVOLVE-BLOCK-END
