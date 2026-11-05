"""Rewritten program for Resource-Constrained Project Scheduling."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time

_START_TIME = None
_TIME_LIMIT = 145.0  # Safety margin before 150s wall-clock kill

def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Heuristic components:
    1. Bottom Level (BL): Longest path from node to end. (Critical Path)
    2. Total Float (TF): Slack; lower float = more critical.
    3. Resource Pressure (RP): Sum of (demand / capacity).
    4. Max Resource Pressure (MRP): Max of (demand / capacity).
    5. Duration: Longer jobs get slight priority to reduce fragmentation.
    6. Empirical Delay: Actual delay from resource contention in a trial SGS run.
    
    Score = w1*BL + w2*Float + w3*RP + w4*MRP + w5*Dur + w6*EmpiricalDelay
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        durations = np.asarray(instance["durations"], dtype=float)
        if len(durations) > 0:
            return durations / (np.max(durations) + 1e-9)
        return np.array([])

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
                
    if len(top_order) < n:
        top_order = list(range(n))

    # Compute Bottom Level (longest path from node to end)
    bottom_level = np.zeros(n)
    for u in reversed(top_order):
        for v in successors[u]:
            candidate = bottom_level[v] + durations[v]
            if candidate > bottom_level[u]:
                bottom_level[u] = candidate
    
    # Compute Top Level (longest path from start to node) = Earliest Start
    top_level = np.zeros(n)
    for u in top_order:
        for p in pred_lists[u]:
            candidate = top_level[p] + durations[p]
            if candidate > top_level[u]:
                top_level[u] = candidate
    
    # Total Float (Slack) = Bottom Level - Top Level - Duration
    total_float = bottom_level - top_level - durations
    total_float = np.maximum(total_float, 0.0)
    
    # Check time limit
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT - 10.0:
        bl_max = np.max(bottom_level) if np.max(bottom_level) > 0 else 1.0
        return bottom_level / bl_max

    # 2. Compute Resource Metrics
    if demands is None or capacities is None:
        bl_max = np.max(bottom_level)
        if bl_max > 0:
            return bottom_level / bl_max
        return bottom_level

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

    # 3. Normalize and Combine static scores
    bl_max = np.max(bottom_level) if np.max(bottom_level) > 0 else 1.0
    rp_max = np.max(resource_pressure) if np.max(resource_pressure) > 0 else 1.0
    mrp_max = np.max(max_resource_pressure) if np.max(max_resource_pressure) > 0 else 1.0
    dur_max = np.max(durations) if np.max(durations) > 0 else 1.0
    tf_max = np.max(total_float) if np.max(total_float) > 0 else 1.0

    bl_norm = bottom_level / bl_max
    rp_norm = resource_pressure / rp_max
    mrp_norm = max_resource_pressure / mrp_max
    dur_norm = durations / dur_max
    float_norm = 1.0 - (total_float / tf_max)

    # Static weights
    w_bl = 0.7
    w_float = 0.25
    w_rp = 0.3
    w_mrp = 0.15
    w_dur = 0.05

    static_scores = w_bl * bl_norm + w_float * float_norm + w_rp * rp_norm + w_mrp * mrp_norm + w_dur * dur_norm

    # 4. Empirical Delay via trial Serial SGS
    # Run a quick serial SGS with static_scores to identify which activities
    # actually get delayed by resource contention.
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT - 15.0:
        # Not enough time for empirical delay; return static scores
        return static_scores

    # Trial SGS: simulate serial schedule generation
    # Track actual start times
    actual_start = np.zeros(n)
    scheduled = np.zeros(n, dtype=bool)
    
    # Resource availability over time: use a list of (end_time, activity) events
    # For each resource, track when it becomes free
    # We'll use a simple approach: maintain resource_free_time for each resource
    
    n_resources = len(capacities_arr)
    # resource_free_time[r] = earliest time resource r is available
    # But this is too simplistic for multiple activities using same resource.
    # Better: maintain a list of (end_time, resource_id, demand) for each resource.
    
    # For efficiency with 30 activities, use a simple event-based approach:
    # For each resource, keep a sorted list of (end_time, demand)
    # When scheduling an activity at time t, find the earliest time when
    # cumulative demand from overlapping activities + new demand <= capacity.
    
    # Simpler: for each resource, maintain a list of (start, end, demand) for
    # already-scheduled activities. To find earliest feasible start, check
    # candidate times: current time, and end times of overlapping activities.
    
    # For 30 activities, this is fast enough.
    
    # For each resource, store list of (start_time, end_time, demand)
    resource_intervals = [[] for _ in range(n_resources)]
    
    # Precompute: for each activity, which resources it uses and how much
    # demand_matrix[i, r] = demand of activity i on resource r
    
    # Serial SGS: repeatedly pick highest-priority eligible activity
    # Eligible: all predecessors scheduled
    remaining = set(range(n))
    completed = set()
    
    while remaining:
        # Find eligible activities (all predecessors in completed)
        eligible = []
        for i in remaining:
            eligible = True
            for p in pred_lists[i]:
                if p not in completed:
                    eligible = False
                    break
            if eligible:
                eligible.append(i)
        
        if not eligible:
            # Deadlock (shouldn't happen in valid DAG)
            break
        
        # Pick highest priority
        best = max(eligible, key=lambda i: static_scores[i])
        
        # Find earliest feasible start time for 'best'
        # Must be >= earliest start from predecessors
        es = top_level[best]
        
        # For each resource, find earliest time when resource is free enough
        earliest_feasible = es
        for r in range(n_resources):
            demand_r = demand_matrix[best, r]
            if demand_r <= 0:
                continue
            if demand_r > capacities_arr[r]:
                # Infeasible (shouldn't happen in valid instance)
                continue
            
            # Find earliest time t >= earliest_feasible where
            # sum of demands of overlapping activities on resource r
            # + demand_r <= capacities_arr[r]
            
            # Candidate times: earliest_feasible, and end times of intervals
            # that start before some candidate and end after.
            
            # Simple approach: iterate over candidate times
            # Candidate times: earliest_feasible, and end times of all intervals
            # on resource r that could overlap.
            
            # For 30 activities, just check all relevant end times.
            candidate_times = set()
            candidate_times.add(earliest_feasible)
            for (s, e, d) in resource_intervals[r]:
                if e > earliest_feasible:
                    candidate_times.add(e)
            
            # Sort candidate times
            candidate_times = sorted(candidate_times)
            
            feasible_time = None
            for t in candidate_times:
                if t < earliest_feasible:
                    continue
                # Check if resource r has enough capacity at time t
                # Sum demands of all intervals on resource r that overlap with
                # [t, t + duration[best]]
                total_demand = demand_r
                for (s, e, d) in resource_intervals[r]:
                    # Check overlap between [s, e] and [t, t + dur]
                    if s < t + durations[best] and e > t:
                        total_demand += d
                if total_demand <= capacities_arr[r] + 1e-9:
                    feasible_time = t
                    break
            
            if feasible_time is None:
                # Shouldn't happen if demand <= capacity
                feasible_time = earliest_feasible
            
            if feasible_time > earliest_feasible:
                earliest_feasible = feasible_time
        
        # Schedule 'best' at earliest_feasible
        actual_start[best] = earliest_feasible
        end_time = earliest_feasible + durations[best]
        
        # Record intervals for each resource
        for r in range(n_resources):
            demand_r = demand_matrix[best, r]
            if demand_r > 0:
                resource_intervals[r].append((earliest_feasible, end_time, demand_r))
        
        remaining.discard(best)
        completed.add(best)
    
    # Compute empirical delay
    empirical_delay = actual_start - top_level
    empirical_delay = np.maximum(empirical_delay, 0.0)
    
    # Normalize empirical delay
    ed_max = np.max(empirical_delay) if np.max(empirical_delay) > 0 else 1.0
    ed_norm = empirical_delay / ed_max
    
    # 5. Final score: combine static + empirical
    # Empirical delay is a strong signal of actual contention
    w_ed = 0.4
    
    scores = static_scores + w_ed * ed_norm
    
    # Check final time limit
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT - 5.0:
        # Return what we have
        pass
    
    return scores
# EVOLVE-BLOCK-END
