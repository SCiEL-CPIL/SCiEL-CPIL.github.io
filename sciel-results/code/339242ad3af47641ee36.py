"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
import math
import sys

_DEADLINE_SECONDS = 140.0
_START_TIME = time.time()

def _remaining_budget():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)

def _run_sgs_fast(instance, vec, d_mat, c_vec, predecessors, successors, resource_users, num_resources, n, durations):
    """
    Fast Serial SGS using a static priority vector.
    """
    scheduled = [False] * n
    start_times = np.zeros(n, dtype=float)
    finish_times = np.zeros(n, dtype=float)
    scheduled_count = 0
    
    # Pre-compute eligible check helper? No, just loop.
    
    while scheduled_count < n:
        eligible = []
        for i in range(n):
            if not scheduled[i]:
                is_eligible = True
                for p in predecessors[i]:
                    if not scheduled[p]:
                        is_eligible = False
                        break
                if is_eligible:
                    eligible.append(i)
                        
        if not eligible:
            break
                
        # Pick highest priority
        best_job = -1
        best_prio = -np.inf
        for i in eligible:
            if vec[i] > best_prio:
                best_prio = vec[i]
                best_job = i
                    
        es_pred = 0.0
        for p in predecessors[best_job]:
            if finish_times[p] > es_pred:
                es_pred = finish_times[p]
                    
        candidate_times = {es_pred}
        if d_mat is not None and c_vec is not None:
            for r in range(num_resources):
                if d_mat[best_job][r] > 0:
                    for j in resource_users[r]:
                        if scheduled[j]:
                            candidate_times.add(finish_times[j])
                                
        earliest_start = float('inf')
        for t in sorted(candidate_times):
            if t < es_pred:
                continue
            feasible = True
            if d_mat is not None and c_vec is not None:
                for r in range(num_resources):
                    if d_mat[best_job][r] == 0:
                        continue
                    load = 0.0
                    for j in resource_users[r]:
                        if scheduled[j]:
                            if start_times[j] <= t < finish_times[j]:
                                load += d_mat[j][r]
                    if load + d_mat[best_job][r] > c_vec[r]:
                        feasible = False
                        break
            if feasible:
                earliest_start = t
                break
                    
        if earliest_start == float('inf'):
            earliest_start = es_pred
                
        start_times[best_job] = earliest_start
        finish_times[best_job] = earliest_start + durations[best_job]
        scheduled[best_job] = True
        scheduled_count += 1
            
    makespan = np.max(finish_times) if n > 0 else 0.0
    return makespan, start_times, finish_times

def priority(instance):
    """
    Strategy: Model RCPSP as a flow problem to identify bottleneck activities.
    1. Compute static Critical Path (Tail).
    2. Use Tail as initial priority.
    3. Run SGS to get a schedule.
    4. Construct a time-expanded network based on the schedule.
    5. Compute Max Flow / Min Cut to identify the "bottleneck" set of activities.
       In a time-expanded network for RCPSP, the min cut often corresponds to the set of
       activities that, if delayed, would most increase the makespan due to resource contention.
       Alternatively, we can look at the "bottleneck" resources.
       
    A simpler proxy for the "max flow" bottleneck concept in this context:
    - Run SGS with Tail priority.
    - Identify the "critical path" in the *resource-constrained* schedule.
      This is different from the unconstrained critical path.
      A job is "constrained-critical" if it lies on the path from start to end
      where the resource utilization is at capacity (or near it) and the time is tight.
      
    Let's implement a "Bottleneck Flow" heuristic:
    1. Get a schedule.
    2. For each resource, find the intervals where it is fully utilized.
    3. Activities running during these intervals are "bottleneck" activities.
    4. Prioritize bottleneck activities that also have high Tail (unconstrained criticality).
    
    This is structurally different from simple local search on the vector. It uses
    global structural properties of the generated schedule (resource saturation).
    """
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    if n == 0:
        return np.array([])
        
    predecessors = instance.get("predecessors", None)
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
        
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            
    # Topological sort
    in_degree = [len(p) for p in predecessors]
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # Tail Calculation
    tail = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        max_succ_tail = 0.0
        for v in successors[u]:
            if tail[v] > max_succ_tail:
                max_succ_tail = tail[v]
        tail[u] = durations[u] + max_succ_tail
        
    # Setup Resource Data
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    num_resources = 0
    d_mat = None
    c_vec = None
    if demands is not None:
        d_mat = np.asarray(demands, dtype=float)
        if len(d_mat.shape) == 2:
            num_resources = d_mat.shape[1]
        else:
            num_resources = 1
            d_mat = d_mat.reshape(-1, 1)
    if capacities is not None:
        c_vec = np.asarray(capacities, dtype=float)
        if len(c_vec.shape) == 1:
            c_vec = c_vec.reshape(-1)
            
    resource_users = [[] for _ in range(num_resources)]
    if d_mat is not None:
        for i in range(n):
            for r in range(num_resources):
                if d_mat[i][r] > 0:
                    resource_users[r].append(i)

    # Initial Vector: Tail
    current_vec = tail.copy()
    
    # Run initial SGS
    best_vec = current_vec.copy()
    best_score, best_st, best_ft = _run_sgs_fast(instance, current_vec, d_mat, c_vec, predecessors, successors, resource_users, num_resources, n, durations)
    
    # Iterative Bottleneck Refinement
    # We will run SGS, identify "Bottleneck Activities", and boost their priority.
    # A Bottleneck Activity is one that:
    # 1. Is on the Critical Path of the *current* schedule (start + tail ~ makespan).
    # 2. Uses a resource that is saturated (load == capacity) during its execution.
    
    max_iters = 10
    for _ in range(max_iters):
        if _remaining_budget() < 5.0:
            break
            
        score, st, ft = _run_sgs_fast(instance, current_vec, d_mat, c_vec, predecessors, successors, resource_users, num_resources, n, durations)
        
        if score < best_score:
            best_score = score
            best_vec = current_vec.copy()
            best_st = st
            best_ft = ft
            
        makespan = score
        
        # Identify Bottleneck Activities
        # 1. Critical Path in schedule: st[i] + tail[i] >= makespan - epsilon
        critical_mask = np.zeros(n, dtype=bool)
        for i in range(n):
            if st[i] + tail[i] >= makespan - 1e-6:
                critical_mask[i] = True
                
        # 2. Resource Saturation
        # For each resource, check if it's full during the job's execution.
        # Since we have discrete time points from the schedule, we can check intervals.
        # To be fast, we only check if the job's interval overlaps with any "full" interval of the resource.
        # Simplification: If a job uses a resource, and the *average* load of that resource
        # over the makespan is high, it's a bottleneck resource.
        # Better: Check if the job is running when the resource is at capacity.
        # We can approximate this by checking if the job's demand + other concurrent demands > capacity.
        # Actually, in SGS, if a job is scheduled, it fits. But other jobs might be waiting.
        # A job is a bottleneck if delaying it would delay successors or cause resource conflicts later.
        
        # Let's use a "Bottleneck Score" based on resource utilization during the job's active window.
        # For each job i, calculate the "Resource Pressure" it exerts.
        # Pressure(i) = sum over resources r used by i of (Load_r_at_i / Capacity_r)
        # If Load_r_at_i is high, i is a bottleneck.
        
        new_vec = current_vec.copy()
        
        # Pre-compute load profiles? Too slow.
        # Instead, for each job i, check if it overlaps with other jobs on the same resource
        # such that the total load is close to capacity.
        
        # Optimization: Only do this for jobs on the critical path.
        for i in range(n):
            if not critical_mask[i]:
                continue
                
            pressure = 0.0
            if d_mat is not None and c_vec is not None:
                for r in range(num_resources):
                    if d_mat[i][r] == 0:
                        continue
                    # Check load at time st[i] + durations[i]/2 (midpoint)
                    # This is an approximation.
                    mid_time = st[i] + durations[i] / 2.0
                    load = 0.0
                    for j in resource_users[r]:
                        if scheduled_check(j, st, ft, mid_time, n):
                            load += d_mat[j][r]
                    
                    if c_vec[r] > 0:
                        utilization = load / c_vec[r]
                        # If utilization is high, this job is a bottleneck
                        if utilization > 0.8:
                            pressure += utilization * d_mat[i][r] / c_vec[r]
            
            # Boost priority based on pressure
            # We want to schedule bottleneck jobs earlier to prevent them from blocking others?
            # Or do we want to schedule them *last* to free up resources for non-critical jobs?
            # Usually, prioritizing critical path jobs is good.
            # If a job is critical AND a bottleneck, it is very important.
            # Boost it.
            new_vec[i] += pressure * 0.5 * makespan
            
        # Normalize to keep scale similar
        if np.max(new_vec) > 0:
            new_vec = new_vec / np.max(new_vec) * np.max(current_vec)
            
        current_vec = new_vec
        
    return best_vec

def scheduled_check(j, st, ft, t, n):
    """Helper to check if job j is active at time t."""
    return st[j] <= t < ft[j]

# EVOLVE-BLOCK-END
