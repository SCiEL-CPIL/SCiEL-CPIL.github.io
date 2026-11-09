"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
import math
from itertools import combinations

_DEADLINE_SECONDS = 140.0
_START_TIME = time.time()

def _remaining_budget():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)

def _compute_slack(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance.get("predecessors", None)
    n = len(durations)
    if n == 0:
        return np.array([])
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
    
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    es = np.zeros(n, dtype=float)
    ef = np.zeros(n, dtype=float)
    for u in topo_order:
        if predecessors[u]:
            es[u] = max(ef[p] for p in predecessors[u])
        ef[u] = es[u] + durations[u]
        
    makespan_no_res = np.max(ef)
    ls = np.zeros(n, dtype=float)
    lf = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if successors[u]:
            lf[u] = min(ls[s] for s in successors[u])
        else:
            lf[u] = makespan_no_res
        ls[u] = lf[u] - durations[u]
        
    slack = ls - es
    return slack

def _run_serial_sgs(instance, priority_vec):
    """
    Serial Schedule Generation Scheme.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    n = len(durations)
    if n == 0:
        return 0.0
        
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
        
    # Pre-compute resource users
    num_resources = 0
    if demands is not None:
        d_mat = np.asarray(demands)
        if len(d_mat.shape) == 2:
            num_resources = d_mat.shape[1]
    
    resource_users = [[] for _ in range(num_resources)]
    if demands is not None:
        d_mat = np.asarray(demands, dtype=float)
        for i in range(n):
            for r in range(num_resources):
                if d_mat[i][r] > 0:
                    resource_users[r].append(i)
                    
    # State
    scheduled = [False] * n
    start_times = np.zeros(n, dtype=float)
    finish_times = np.zeros(n, dtype=float)
    
    scheduled_count = 0
    while scheduled_count < n:
        best_job = -1
        best_prio = -np.inf
        
        for i in range(n):
            if not scheduled[i]:
                eligible = True
                for p in predecessors[i]:
                    if not scheduled[p]:
                        eligible = False
                        break
                if eligible:
                    if priority_vec[i] > best_prio:
                        best_prio = priority_vec[i]
                        best_job = i
                        
        if best_job == -1:
            break
            
        es_pred = 0.0
        for p in predecessors[best_job]:
            if finish_times[p] > es_pred:
                es_pred = finish_times[p]
                
        candidate_times = {es_pred}
        if demands is not None and capacities is not None:
            d_mat = np.asarray(demands, dtype=float)
            c_vec = np.asarray(capacities, dtype=float)
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
            if demands is not None and capacities is not None:
                d_mat = np.asarray(demands, dtype=float)
                c_vec = np.asarray(capacities, dtype=float)
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
    return makespan

def priority(instance):
    """
    Priority function using Simulated Annealing on the priority vector.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    if n == 0:
        return np.array([])
        
    np.random.seed(42)
    
    # Initial Priorities based on Slack and Duration
    slack = _compute_slack(instance)
    # Normalize slack so smaller is higher priority (or just use negative slack)
    p_slack = -slack
    
    # Scale duration to similar magnitude as slack for hybrid
    if np.max(durations) > 0:
        p_dur = durations / np.max(durations) * np.max(np.abs(p_slack)) if np.max(np.abs(p_slack)) > 0 else durations
    else:
        p_dur = np.zeros(n)
        
    initial_vec = p_slack + p_dur * 0.5
    
    best_vec = initial_vec.copy()
    best_score = _run_serial_sgs(instance, initial_vec)
    
    current_vec = initial_vec.copy()
    current_score = best_score
    
    T = 1.0
    cooling = 0.99
    min_T = 1e-3
    
    while _remaining_budget() > 5.0:
        if T < min_T:
            break
            
        # Perturbation: Pick 2 random indices and add noise
        # Or pick 1 index and add noise. With n=30, picking 2 is fine.
        k = np.random.randint(1, 3)
        indices = np.random.choice(n, size=k, replace=False)
        noise = np.random.randn(k) * 0.5
        
        new_vec = current_vec.copy()
        new_vec[indices] += noise
        
        new_score = _run_serial_sgs(instance, new_vec)
        
        delta = new_score - current_score
        if delta < 0:
            current_vec = new_vec
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_vec = new_vec.copy()
        else:
            # Accept with probability exp(-delta / T)
            if T > 1e-9:
                prob = math.exp(-delta / T)
                if np.random.rand() < prob:
                    current_vec = new_vec
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_vec = new_vec.copy()
                        
        T *= cooling
        
    return best_vec

# EVOLVE-BLOCK-END
