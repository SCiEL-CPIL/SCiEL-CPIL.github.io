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
    Strategy: Multi-Objective Pareto Frontier Search over static priority vectors.
    Instead of refining a single vector, we generate a diverse set of candidate
    priority vectors based on different structural metrics (Tail, Rank, Resource
    Weight, etc.), evaluate each with a fast SGS, and return the vector that
    produces the best makespan. This explores a wider solution space of
    scheduling philosophies than local refinement.
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
                
    # Tail Calculation (Backward Pass)
    tail = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        max_succ_tail = 0.0
        for v in successors[u]:
            if tail[v] > max_succ_tail:
                max_succ_tail = tail[v]
        tail[u] = durations[u] + max_succ_tail
        
    # Rank Calculation (Forward Pass)
    rank = np.zeros(n, dtype=float)
    for u in topo_order:
        max_pred_rank = 0.0
        for p in predecessors[u]:
            if rank[p] > max_pred_rank:
                max_pred_rank = rank[p]
        rank[u] = durations[u] + max_pred_rank
        
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
                
    # Generate Candidate Vectors
    candidates = []
    
    # 1. Tail (Standard Critical Path)
    candidates.append(tail.copy())
    
    # 2. Rank (Forward Critical Path)
    candidates.append(rank.copy())
    
    # 3. Resource-Weighted Tail
    # Prioritize activities that use scarce resources heavily
    if d_mat is not None and c_vec is not None:
        resource_pressure = np.zeros(n, dtype=float)
        for i in range(n):
            max_pressure = 0.0
            for r in range(num_resources):
                if d_mat[i][r] > 0 and c_vec[r] > 0:
                    # Higher demand relative to capacity means more pressure
                    p = d_mat[i][r] / c_vec[r]
                    if p > max_pressure:
                        max_pressure = p
            resource_pressure[i] = max_pressure
            
        # Combine Tail and Resource Pressure
        # Normalize both to [0,1]
        max_tail = np.max(tail) if np.max(tail) > 0 else 1.0
        max_pressure = np.max(resource_pressure) if np.max(resource_pressure) > 0 else 1.0
        
        # Weighted sum
        w1 = 0.5
        w2 = 0.5
        vec_rw = w1 * (tail / max_tail) + w2 * (resource_pressure / max_pressure)
        candidates.append(vec_rw)
        
        # 4. Pure Resource Pressure
        candidates.append(resource_pressure)
        
        # 5. Tail * (1 + Resource Pressure)
        vec_tp = tail * (1.0 + resource_pressure)
        candidates.append(vec_tp)
        
    # 6. Random Perturbation of Tail (to escape local optima of static metrics)
    # Use a fixed seed for reproducibility within the run, but different from tail
    rng = np.random.RandomState(42)
    noise = rng.uniform(0, 0.1 * np.max(tail), size=n)
    vec_noisy = tail + noise
    candidates.append(vec_noisy)
    
    # 7. Reverse Tail (Lowest Tail first? Usually bad, but check Pareto)
    # Actually, let's try "Minimum Tail" as a heuristic for some problems
    candidates.append(-tail) # This will schedule lowest tail first? No, SGS picks highest priority.
    # So -tail means highest priority is lowest tail.
    
    # Evaluate Candidates
    best_vec = candidates[0]
    best_score = float('inf')
    best_st = None
    best_ft = None
    
    for vec in candidates:
        if _remaining_budget() < 5.0:
            break
            
        score, st, ft = _run_sgs_fast(instance, vec, d_mat, c_vec, predecessors, successors, resource_users, num_resources, n, durations)
        
        if score < best_score:
            best_score = score
            best_vec = vec.copy()
            best_st = st
            best_ft = ft
            
    # If we have time, try a few more random perturbations
    if _remaining_budget() > 10.0:
        for _ in range(5):
            if _remaining_budget() < 5.0:
                break
            # Perturb the best vector found so far
            noise = rng.uniform(0, 0.05 * np.max(best_vec), size=n)
            vec_pert = best_vec + noise
            score, st, ft = _run_sgs_fast(instance, vec_pert, d_mat, c_vec, predecessors, successors, resource_users, num_resources, n, durations)
            if score < best_score:
                best_score = score
                best_vec = vec_pert.copy()
                best_st = st
                best_ft = ft

    return best_vec

# EVOLVE-BLOCK-END
