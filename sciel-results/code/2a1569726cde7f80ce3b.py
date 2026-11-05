"""Rewritten program for Resource-Constrained Project Scheduling."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time

_TIME_LIMIT = 148.0
_START_TIME = None

def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Direct approach: 
    - Critical Path Method (CPM) for float/slack
    - Resource pressure metrics
    - Bottom level (longest path to project end)
    - Combined with careful weighting
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Emergency fallback
        durations = np.asarray(instance["durations"], dtype=float)
        return durations / (np.max(durations) + 1e-9)

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if n == 0:
        return np.array([])

    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    preds = instance.get("predecessors", [None] * n)

    # Build graph
    pred_lists = []
    for p in preds:
        if p is None:
            pred_lists.append([])
        elif isinstance(p, (list, tuple, np.ndarray)):
            pred_lists.append([x for x in p if x is not None and 0 <= x < n])
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
                
    if len(top_order) < n:
        top_order = list(range(n))

    # Forward pass: ES, EF
    es = np.zeros(n)
    ef = np.zeros(n)
    for u in top_order:
        ef[u] = es[u] + durations[u]
        for v in successors[u]:
            if es[v] < ef[u]:
                es[v] = ef[u]
    
    # Backward pass: LS, LF
    project_end = np.max(ef) if n > 0 else 0.0
    lf = np.full(n, project_end)
    ls = np.zeros(n)
    for u in reversed(top_order):
        ls[u] = lf[u] - durations[u]
        for v in successors[u]:
            if lf[u] > ls[v]:
                lf[u] = ls[v]
    
    # Total Float (Slack)
    total_float = ls - es
    
    # Bottom Level: longest path from activity to project end
    bottom_level = np.zeros(n)
    for u in reversed(top_order):
        max_succ = 0.0
        for v in successors[u]:
            candidate = durations[v] + bottom_level[v]
            if candidate > max_succ:
                max_succ = candidate
        bottom_level[u] = max_succ

    # Resource Metrics
    if demands is None or capacities is None:
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
        
    if demand_matrix.shape[1] != len(capacities_arr):
        new_dim = min(demand_matrix.shape[1], len(capacities_arr))
        demand_matrix = demand_matrix[:, :new_dim]
        capacities_arr = capacities_arr[:new_dim]
        
    safe_caps = np.where(capacities_arr > 0, capacities_arr, 1.0)
    
    # Resource pressure metrics
    ratios = demand_matrix / safe_caps
    resource_pressure = np.sum(ratios, axis=1)
    max_resource_pressure = np.max(ratios, axis=1)

    # Simulate a greedy resource-feasible schedule to identify contention
    # This helps identify which activities actually cause bottlenecks
    if time.time() - _START_TIME < _TIME_LIMIT - 5.0:
        # Use float-based priority for the simulation
        sim_float_max = np.max(total_float) if np.max(total_float) > 0 else 1.0
        sim_priority = -total_float / sim_float_max
        
        # Simulate serial SGS
        scheduled = [False] * n
        start_times = [0.0] * n
        end_times = [0.0] * n
        resource_usage = np.zeros(len(capacities_arr))
        
        # Track when resources become available
        resource_free_times = np.zeros(len(capacities_arr))
        
        remaining = list(range(n))
        
        while remaining:
            # Find eligible activities (all predecessors scheduled)
            eligible = []
            for i in remaining:
                all_preds_done = True
                for p in pred_lists[i]:
                    if not scheduled[p]:
                        all_preds_done = False
                        break
                if all_preds_done:
                    eligible.append(i)
            
            if not eligible:
                # Break cycle - schedule the one with highest priority
                eligible = [max(remaining, key=lambda x: sim_priority[x])]
            
            # Pick the highest priority eligible activity
            best = max(eligible, key=lambda x: sim_priority[x])
            
            # Determine earliest start time considering resource availability
            # An activity can start when all predecessors are done AND resources are available
            earliest_start = 0.0
            for p in pred_lists[best]:
                if end_times[p] > earliest_start:
                    earliest_start = end_times[p]
            
            # Find the earliest time when all required resources are available
            # This is a simplification - we assume resources are released at end_times
            # For a more accurate simulation, we'd need to track resource release times
            
            # For now, use a simple approximation: start at earliest_start
            # and assume resources are available (this is optimistic)
            start_time = earliest_start
            
            # Schedule the activity
            scheduled[best] = True
            start_times[best] = start_time
            end_times[best] = start_time + durations[best]
            
            # Update resource usage (this is a simplification)
            # In a real simulation, we'd need to track when resources are freed
            resource_usage += demand_matrix[best]
            
            remaining.remove(best)
        
        # Identify activities that are on the critical path AND have high resource demand
        # Critical activities: those with low total float
        # Resource bottleneck activities: those with high demand-to-capacity ratio
        
        # Compute a contention score: how much each activity contributes to resource contention
        # We approximate this by looking at the resource pressure and whether the activity
        # is on a path that delays the project
        
        # For activities with high resource pressure and low float, they are likely bottlenecks
        contention_score = resource_pressure * (1.0 - total_float / (np.max(total_float) + 1e-9))
        contention_max = np.max(contention_score) if np.max(contention_score) > 0 else 1.0
        contention_norm = contention_score / contention_max

    # Normalize
    float_min = np.min(total_float)
    float_range = np.max(total_float) - float_min
    if float_range > 0:
        float_norm = 1.0 - (total_float - float_min) / float_range
    else:
        float_norm = np.ones(n)
    
    rp_max = np.max(resource_pressure) if np.max(resource_pressure) > 0 else 1.0
    mrp_max = np.max(max_resource_pressure) if np.max(max_resource_pressure) > 0 else 1.0
    bl_max = np.max(bottom_level) if np.max(bottom_level) > 0 else 1.0
    dur_max = np.max(durations) if np.max(durations) > 0 else 1.0

    rp_norm = resource_pressure / rp_max
    mrp_norm = max_resource_pressure / mrp_max
    bl_norm = bottom_level / bl_max
    dur_norm = durations / dur_max

    # Weights: Float primary, bottom level secondary, resource metrics tertiary
    # Add contention as a significant factor
    w_float = 1.0
    w_bl = 0.3
    w_rp = 0.2
    w_mrp = 0.1
    w_dur = 0.05
    w_contention = 0.35

    scores = w_float * float_norm + w_bl * bl_norm + w_rp * rp_norm + w_mrp * mrp_norm + w_dur * dur_norm
    if time.time() - _START_TIME < _TIME_LIMIT - 5.0:
        scores += w_contention * contention_norm

    return scores
# EVOLVE-BLOCK-END
