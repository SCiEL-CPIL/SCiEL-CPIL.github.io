"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
import math
import sys
import random

_DEADLINE_SECONDS = 140.0
_START_TIME = time.time()

def _remaining_budget():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)

def _run_sgs_fast(instance, vec, d_mat, c_vec, predecessors, successors, resource_users, num_resources, n, durations):
    """
    Fast Serial SGS using a static priority vector.
    Returns makespan.
    """
    scheduled = [False] * n
    start_times = np.zeros(n, dtype=float)
    finish_times = np.zeros(n, dtype=float)
    scheduled_count = 0
    
    # Pre-compute eligible check helper? No, just loop.
    # Optimization: Keep a list of unscheduled nodes.
    unscheduled = set(range(n))
    
    while unscheduled:
        eligible = []
        for i in unscheduled:
            is_eligible = True
            for p in predecessors[i]:
                if not scheduled[p]:
                    is_eligible = False
                    break
            if is_eligible:
                eligible.append(i)
                
        if not eligible:
            # Should not happen if DAG is valid, but break to avoid infinite loop
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
                    if load + d_mat[best_job][r] > c_vec[r] + 1e-9:
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
        unscheduled.remove(best_job)
        scheduled_count += 1
        
    makespan = np.max(finish_times) if n > 0 else 0.0
    return makespan

def priority(instance):
    """
    Strategy: Simulated Annealing on Priority Vectors.
    1. Initialize with Tail (Critical Path) priority.
    2. Define a neighborhood: perturb a random subset of priorities.
    3. Objective: Minimize Makespan from Serial SGS.
    4. Accept worse solutions with probability exp(-delta/T) to escape local minima.
    5. Run until time budget is exhausted.
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
    current_score = _run_sgs_fast(instance, current_vec, d_mat, c_vec, predecessors, successors, resource_users, num_resources, n, durations)
    
    best_vec = current_vec.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    # Start temperature high enough to accept significant changes
    T_start = max(1.0, current_score * 0.1) 
    T_end = 1e-6
    # Number of iterations is dynamic based on time
    # We will run until time is up.
    
    # Use a fixed seed for reproducibility in SA randomness? 
    # The task evaluator likely runs on different instances, so randomness is fine.
    # However, to be safe against specific seed issues, we just use standard random.
    
    # Pre-calculate random indices for speed? No, n=30 is small.
    
    while _remaining_budget() > 5.0:
        # Cooling: Geometric cooling is standard, but we don't know total iterations.
        # We can use a time-based cooling or just a fixed number of steps per "epoch".
        # Let's do a batch of 100 steps, then re-evaluate T based on remaining time?
        # Or simpler: Fixed geometric cooling per step, but restart if stuck?
        # Let's try: T = T_start * (T_end/T_start)^(step/max_steps_estimate)
        # Since max_steps is unknown, let's use a simple decay factor per step.
        # Decay factor 0.999 per step?
        
        # Better: Estimate remaining steps. 
        # If we assume ~1ms per step, and we have X seconds left, we have X*1000 steps.
        # This is risky if SGS is slower.
        
        # Alternative: Just use a fixed decay rate that works well for ~10k-100k steps.
        # Let's use alpha = 0.995.
        # If we run for 140s, and each step takes ~1ms, that's 140k steps.
        # T_final = T_start * 0.995^140000 = T_start * exp(-700) ~ 0. This is too fast.
        
        # Let's use a slower decay.
        # We want T to stay high for a while.
        # Let's try alpha = 0.9999.
        # 0.9999^10000 ~ 0.36.
        
        # Actually, let's just run a fixed number of iterations per loop iteration of the while loop?
        # No, the while loop is the SA loop.
        
        # Let's use a dynamic temperature based on remaining time to ensure we reach low T.
        # But we don't know total time.
        
        # Let's stick to a robust heuristic:
        # Perturb 1-3 random activities.
        
        new_vec = current_vec.copy()
        
        # Number of perturbations
        num_pert = random.randint(1, 3)
        indices = random.sample(range(n), min(num_pert, n))
        
        for idx in indices:
            # Add noise proportional to current scale
            noise = random.gauss(0, current_score * 0.05)
            new_vec[idx] += noise
            # Ensure positive? Tail is positive. Noise can make it negative.
            # Negative priority is fine in SGS (it just gets picked last).
            
        new_score = _run_sgs_fast(instance, new_vec, d_mat, c_vec, predecessors, successors, resource_users, num_resources, n, durations)
        
        delta = new_score - current_score
        
        # Accept if better
        if delta < 0:
            current_vec = new_vec
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_vec = new_vec.copy()
        else:
            # Accept if worse with probability exp(-delta/T)
            # We need T.
            # Let's use a time-based temperature.
            # Elapsed time t. Total time T_total = 140.
            # T = T_start * exp(-lambda * t)
            # We want T to drop significantly.
            # Let's say lambda = 0.05.
            # t=140 -> exp(-7) ~ 0.0009.
            elapsed = time.time() - _START_TIME
            T = T_start * math.exp(-0.02 * elapsed)
            
            if random.random() < math.exp(-delta / max(T, 1e-6)):
                current_vec = new_vec
                current_score = new_score
                
    return best_vec

# EVOLVE-BLOCK-END
