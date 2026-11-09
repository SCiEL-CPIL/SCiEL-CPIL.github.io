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

def _compute_cp_features(instance):
    """
    Compute static Critical Path Method (CPM) features:
    - tail: Longest path from node to sink
    - head: Longest path from source to node
    - total_float: head + duration + tail - critical_path_length
    - resource_pressure: Sum of (demand/capacity) for each resource used
    """
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    predecessors = instance.get("predecessors", None)
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
        
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            
    # Topological Sort
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
                
    # Head (Longest path from source)
    head = np.zeros(n, dtype=float)
    for u in topo_order:
        max_pred_head = 0.0
        for p in predecessors[u]:
            if head[p] + durations[p] > max_pred_head:
                max_pred_head = head[p] + durations[p]
        head[u] = max_pred_head
        
    # Tail (Longest path to sink)
    tail = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        max_succ_tail = 0.0
        for v in successors[u]:
            if tail[v] + durations[v] > max_succ_tail:
                max_succ_tail = tail[v] + durations[v]
        tail[u] = max_succ_tail
        
    # Critical Path Length
    cp_len = 0.0
    for i in range(n):
        if head[i] + durations[i] + tail[i] > cp_len:
            cp_len = head[i] + durations[i] + tail[i]
            
    # Total Float
    total_float = cp_len - (head + durations + tail)
    total_float = np.maximum(total_float, 0.0)
    
    # Resource Pressure
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    resource_pressure = np.zeros(n, dtype=float)
    if demands is not None and capacities is not None:
        d_mat = np.asarray(demands, dtype=float)
        c_vec = np.asarray(capacities, dtype=float)
        if len(d_mat.shape) == 1:
            d_mat = d_mat.reshape(-1, 1)
        if len(c_vec.shape) == 1:
            c_vec = c_vec.reshape(-1)
            
        for i in range(n):
            for r in range(d_mat.shape[1]):
                if c_vec[r] > 0:
                    resource_pressure[i] += (d_mat[i][r] / c_vec[r])
                    
    return head, tail, total_float, resource_pressure

def _run_sgs(instance, priority_vec):
    """
    Serial Schedule Generation Scheme using the provided static priority vector.
    Returns the makespan.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    if n == 0:
        return 0.0
        
    predecessors = instance.get("predecessors", None)
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
        
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
            
    # Pre-compute resource users for efficiency
    resource_users = [[] for _ in range(num_resources)]
    if d_mat is not None:
        for i in range(n):
            for r in range(num_resources):
                if d_mat[i][r] > 0:
                    resource_users[r].append(i)

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
            
        # Find highest priority eligible activity
        best_job = -1
        best_prio = -np.inf
        for i in eligible:
            if priority_vec[i] > best_prio:
                best_prio = priority_vec[i]
                best_job = i
                
        if best_job == -1:
            break
            
        # Earliest start based on predecessors
        es_pred = 0.0
        for p in predecessors[best_job]:
            if finish_times[p] > es_pred:
                es_pred = finish_times[p]
                
        # Candidate times for resource check
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
    return makespan

def priority(instance):
    """
    Generates a static priority vector using a composite heuristic of CPM features.
    Weights are tuned to balance criticality (tail/head), slack (float), and resource contention.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    if n == 0:
        return np.array([])
        
    head, tail, total_float, resource_pressure = _compute_cp_features(instance)
    
    # Normalize features to [0, 1] range for stable weighting
    def norm(x):
        mn = np.min(x)
        mx = np.max(x)
        if mx - mn < 1e-9:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
        
    n_head = norm(head)
    n_tail = norm(tail)
    n_float = norm(total_float)
    n_res = norm(resource_pressure)
    
    # Weights:
    # Tail is the most important indicator of criticality.
    # Head helps with early scheduling.
    # Float (inverse) helps identify non-critical jobs to delay.
    # Resource pressure helps identify jobs that consume scarce resources.
    
    # Strategy: High Tail, High Head, Low Float, High Resource Pressure -> High Priority
    # Note: We want to MINIMIZE makespan, so we prioritize jobs that are "urgent".
    # Urgent = High Tail (close to end), Low Float (no slack), High Resource Demand (hard to schedule later).
    
    # Let's try a weighted sum.
    # w_tail = 1.0
    # w_head = 0.2
    # w_float_inv = 0.5  (Inverse of normalized float)
    # w_res = 0.3
    
    # Inverse float: 1 - n_float
    priority_vec = 1.0 * n_tail + 0.2 * n_head + 0.5 * (1.0 - n_float) + 0.3 * n_res
    
    # Optional: Small local search to tweak weights?
    # Given the time constraint and that the heuristic is already strong, 
    # a simple grid search on weights might be beneficial but risky if it takes too long.
    # Let's stick to the static composite score for reliability and speed, 
    # as it captures the main structural aspects of the RCPSP.
    
    return priority_vec

# EVOLVE-BLOCK-END
