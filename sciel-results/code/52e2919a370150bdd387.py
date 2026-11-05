"""Rewritten program for Resource-Constrained Project Scheduling."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time

_TIME_LIMIT = 145.0  # Safety margin before 150s wall-clock kill
_START_TIME = None

def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Heuristic: CPM-based Total Float (slack) as primary driver,
    combined with resource contention metrics.
    
    Score = w1*Float + w2*RP + w3*MRP + w4*Dur
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Emergency fallback: return simple duration-based scores
        durations = np.asarray(instance["durations"], dtype=float)
        return durations / (np.max(durations) + 1e-9)

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if n == 0:
        return np.array([])

    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    preds = instance.get("predecessors", [None] * n)

    # 1. Build graph and compute CPM values
    pred_lists = []
    for p in preds:
        if p is None:
            pred_lists.append([])
        elif isinstance(p, (list, tuple, np.ndarray)):
            pred_lists.append([x for x in p if x is not None])
        else:
            pred_lists.append([p])
            
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in pred_lists[i]:
            if 0 <= p < n:
                successors[p].append(i)
                in_degree[i] += 1
            
    # Topological Sort
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    top_order = []
    while queue:
        u = queue.popleft()
        top_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # If cycle exists, fallback to index order
    if len(top_order) < n:
        top_order = list(range(n))

    # Compute Early Start (ES) and Early Finish (EF) - forward pass
    es = np.zeros(n)
    ef = np.zeros(n)
    for u in top_order:
        for v in successors[u]:
            if es[v] < ef[u]:
                es[v] = ef[u]
        ef[u] = es[u] + durations[u]
    
    # Compute Late Finish (LF) and Late Start (LS) - backward pass
    project_end = np.max(ef) if n > 0 else 0.0
    lf = np.full(n, project_end)
    ls = np.zeros(n)
    for u in reversed(top_order):
        for v in successors[u]:
            if lf[u] > ls[v]:
                lf[u] = ls[v]
        ls[u] = lf[u] - durations[u]
    
    # Total Float (Slack) = LS - ES = LF - EF
    total_float = ls - es
    
    # Bottom level (longest path from node to end)
    bottom_level = np.zeros(n)
    for u in reversed(top_order):
        for v in successors[u]:
            candidate = bottom_level[v] + durations[v]
            if candidate > bottom_level[u]:
                bottom_level[u] = candidate

    # 2. Compute Resource Metrics
    if demands is None or capacities is None:
        # Fallback: Just negative float (lower float = higher priority)
        float_max = np.max(total_float) if np.max(total_float) > 0 else 1.0
        return -total_float / float_max

    # Process demands into matrix
    if isinstance(demands, list) and len(demands) > 0:
        if isinstance(demands[0], dict):
            n_resources = len(capacities)
            demand_matrix = np.zeros((n, n_resources), dtype=float)
            for i, d in enumerate(demands):
                for r, val in d.items():
                    if r < n_resources:
                        demand_matrix[i, r] = val
        elif isinstance(demands[0], (list, tuple, np.ndarray)):
            demand_matrix = np.asarray(demands, dtype=float)
            if demand_matrix.shape[0] != n:
                try:
                    demand_matrix = demand_matrix.reshape(n, -1)
                except:
                    demand_matrix = np.zeros((n, 1))
        else:
            demand_matrix = np.zeros((n, 1), dtype=float)
    else:
        demand_matrix = np.zeros((n, 1), dtype=float)
        
    capacities_arr = np.asarray(capacities, dtype=float)
    if len(capacities_arr) == 0:
        capacities_arr = np.array([1.0])
        
    # Ensure dimensions match
    if demand_matrix.shape[1] != len(capacities_arr):
        new_dim = min(demand_matrix.shape[1], len(capacities_arr))
        demand_matrix = demand_matrix[:, :new_dim]
        capacities_arr = capacities_arr[:new_dim]
        
    safe_caps = np.where(capacities_arr > 0, capacities_arr, 1.0)
    
    # Resource Pressure: Sum of ratios
    resource_pressure = np.sum(demand_matrix / safe_caps, axis=1)
    # Max Resource Pressure: Max of ratios
    max_resource_pressure = np.max(demand_matrix / safe_caps, axis=1)

    # 3. Normalize and Combine
    # Primary: Negative Total Float (activities with zero/low slack are critical)
    # Secondary: Resource pressure to handle contention
    
    float_min = np.min(total_float)
    float_range = np.max(total_float) - float_min
    if float_range > 0:
        # Normalize float so that 0 (critical) gets high score, max float gets low
        float_norm = 1.0 - (total_float - float_min) / float_range
    else:
        float_norm = np.ones(n)
    
    rp_max = np.max(resource_pressure) if np.max(resource_pressure) > 0 else 1.0
    mrp_max = np.max(max_resource_pressure) if np.max(max_resource_pressure) > 0 else 1.0
    dur_max = np.max(durations) if np.max(durations) > 0 else 1.0

    rp_norm = resource_pressure / rp_max
    mrp_norm = max_resource_pressure / mrp_max
    dur_norm = durations / dur_max

    # Weights: Float is the primary CPM-based driver
    # Resource metrics handle contention
    # Tune: emphasize critical path more, reduce duration noise
    w_float = 1.0
    w_rp = 0.4
    w_mrp = 0.25
    w_dur = 0.0

    scores = w_float * float_norm + w_rp * rp_norm + w_mrp * mrp_norm + w_dur * dur_norm
    
    # Add a small tie-breaker: among activities with similar float, prefer those
    # with higher bottom level (longer path to completion) to break ties in favor
    # of activities that are on or near the critical path
    bl_max = np.max(bottom_level) if np.max(bottom_level) > 0 else 1.0
    bl_norm = bottom_level / bl_max
    
    # Blend in a small amount of bottom level as a secondary tie-breaker
    scores = scores + 0.05 * bl_norm

    return scores
# EVOLVE-BLOCK-END
