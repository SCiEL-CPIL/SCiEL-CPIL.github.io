"""
Strategy: Critical Path Density + Critical Ratio
Combines the bottleneck critical ratio (downstream_workload / own_time) with
a critical path density metric (own_time / longest_downstream_path) and
immediate successor count to prioritize tasks that are both bottlenecks in the
critical path and high-fanout branching points.

We also include a lightweight local search to refine the priority vector if time permits.
"""
import numpy as np
import time
from collections import deque

def _compute_downstream_workload(times, predecessors, n):
    """
    Compute the total workload (sum of times) for each task and all its successors,
    the longest downstream path length, and the count of immediate successors.
    Returns (downstream_sum, longest_path, succ_count).
    """
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Topological sort
    in_degree = [len(predecessors[i]) for i in range(n)]
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(topo_order) < n:
        for i in range(n):
            if i not in topo_order:
                topo_order.append(i)
                
    # Reverse topological order to calculate downstream sums and longest path
    downstream_sum = np.zeros(n, dtype=float)
    longest_path = np.zeros(n, dtype=float)
    succ_count = np.zeros(n, dtype=int)
    
    for u in reversed(topo_order):
        downstream_sum[u] = times[u]
        longest_path[u] = times[u]
        succ_count[u] = len(successors[u])
        for v in successors[u]:
            downstream_sum[u] += downstream_sum[v]
            longest_path[u] = max(longest_path[u], times[u] + longest_path[v])
            
    return downstream_sum, longest_path, succ_count

def _simulate_greedy(priority_scores, times, predecessors, cycle_time):
    """
    Simulates the greedy assignment process.
    Returns the number of stations used.
    Optimized with precomputed predecessor lists and list-based remaining tracking.
    """
    n = len(times)
    assigned = [False] * n
    station_load = 0.0
    num_stations = 1
    remaining = list(range(n))
    remaining_set = set(remaining)
    
    # Precompute predecessor lists for faster eligibility
    pred_lists = [list(p) for p in predecessors]
    
    while remaining:
        best_task_idx = -1
        best_priority = -np.inf
        
        # Find the highest priority task that is eligible and fits
        for i in remaining:
            # Check eligibility: all predecessors must be assigned
            eligible = True
            for p in pred_lists[i]:
                if not assigned[p]:
                    eligible = False
                    break
            
            if not eligible:
                continue
            
            # Check if it fits in the current station
            if station_load + times[i] <= cycle_time:
                if priority_scores[i] > best_priority:
                    best_priority = priority_scores[i]
                    best_task_idx = i
        
        if best_task_idx != -1:
            # Assign the task
            assigned[best_task_idx] = True
            station_load += times[best_task_idx]
            remaining.remove(best_task_idx)
        else:
            # No task fits in the current station. Start a new station.
            num_stations += 1
            station_load = 0.0
            
            # In a new station, find the highest priority eligible task that fits
            best_task_idx = -1
            best_priority = -np.inf
            
            for i in remaining:
                eligible = True
                for p in pred_lists[i]:
                    if not assigned[p]:
                        eligible = False
                        break
                
                if not eligible:
                    continue
                
                if times[i] <= cycle_time:
                    if priority_scores[i] > best_priority:
                        best_priority = priority_scores[i]
                        best_task_idx = i
            
            if best_task_idx != -1:
                assigned[best_task_idx] = True
                station_load += times[best_task_idx]
                remaining.remove(best_task_idx)
            else:
                # Should not happen in feasible instances
                break
                
    return num_stations

def priority(instance):
    """
    Computes priority scores combining Critical Ratio and Critical Path Density.
    Critical Ratio = Downstream_Workload / Own_Time
    Critical Path Density = Own_Time / Longest_Downstream_Path (higher when task dominates its path)
    Successor Count = number of immediate successors (branching factor)
    Combined score emphasizes tasks that are both path bottlenecks and high-fanout nodes.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer before 150s limit
    
    # 1. Compute downstream workload, longest path, and successor count
    downstream_workload, longest_path, succ_count = _compute_downstream_workload(times, predecessors, n)
    
    # 2. Compute Critical Ratio and Critical Path Density
    safe_times = np.where(times == 0, 1e-9, times)
    safe_longest = np.where(longest_path == 0, 1e-9, longest_path)
    
    critical_ratio = downstream_workload / safe_times
    # Critical path density: higher when the task's own time is a large fraction of its longest downstream path
    path_density = times / safe_longest
    
    # 3. Combine metrics: weighted sum with normalization
    # Normalize each component to [0, 1] range for fair combination
    cr_min, cr_max = np.min(critical_ratio), np.max(critical_ratio)
    pd_min, pd_max = np.min(path_density), np.max(path_density)
    sc_min, sc_max = float(np.min(succ_count)), float(np.max(succ_count))
    
    if cr_max > cr_min:
        cr_norm = (critical_ratio - cr_min) / (cr_max - cr_min)
    else:
        cr_norm = np.zeros(n)
    
    if pd_max > pd_min:
        pd_norm = (path_density - pd_min) / (pd_max - pd_min)
    else:
        pd_norm = np.zeros(n)
    
    if sc_max > sc_min:
        sc_norm = (succ_count - sc_min) / (sc_max - sc_min)
    else:
        sc_norm = np.zeros(n)
    
    # Weighted combination: emphasize critical ratio most, then path density, then fanout
    best_priority_scores = 0.5 * cr_norm + 0.35 * pd_norm + 0.15 * sc_norm
    best_stations = _simulate_greedy(best_priority_scores, times, predecessors, cycle_time)
    
    # 4. Local Search / Refinement
    # Try multiple weight combinations to find the best structural signal
    rng = np.random.default_rng(42)
    
    # Try different weight combinations for the composite score
    weight_combos = [
        (0.5, 0.35, 0.15),  # default
        (0.6, 0.3, 0.1),
        (0.45, 0.4, 0.15),
        (0.55, 0.3, 0.15),
        (0.4, 0.45, 0.15),
        (0.65, 0.25, 0.1),
        (0.5, 0.4, 0.1),
        (0.7, 0.2, 0.1),
    ]
    
    for w_cr, w_pd, w_sc in weight_combos:
        if time.time() > deadline - 5.0:
            break
        
        candidate_scores = w_cr * cr_norm + w_pd * pd_norm + w_sc * sc_norm
        stations = _simulate_greedy(candidate_scores, times, predecessors, cycle_time)
        
        if stations < best_stations:
            best_stations = stations
            best_priority_scores = candidate_scores.copy()
            
            total_time = np.sum(times)
            lower_bound = int(np.ceil(total_time / cycle_time))
            if best_stations == lower_bound:
                break
    
    # Random perturbation search
    iterations = 0
    max_iterations = 80
    while time.time() < deadline - 2.0 and iterations < max_iterations:
        iterations += 1
        noise = rng.uniform(-0.05, 0.05, n)
        candidate_scores = best_priority_scores + noise
        
        stations = _simulate_greedy(candidate_scores, times, predecessors, cycle_time)
        
        if stations < best_stations:
            best_stations = stations
            best_priority_scores = candidate_scores.copy()
            
            total_time = np.sum(times)
            lower_bound = int(np.ceil(total_time / cycle_time))
            if best_stations == lower_bound:
                break
    
    # Pairwise swap local search
    order = np.argsort(-best_priority_scores)
    n_swaps = 0
    max_swaps = 300
    
    while time.time() < deadline - 2.0 and n_swaps < max_swaps:
        n_swaps += 1
        
        i_idx = rng.integers(0, n)
        rank_i = np.where(order == i_idx)[0][0]
        
        # Pick a neighbor in rank space (within a small window)
        if rank_i > 0:
            rank_j = rank_i - 1
        elif rank_i < n - 1:
            rank_j = rank_i + 1
        else:
            continue
        
        j_idx = order[rank_j]
        
        if i_idx == j_idx:
            continue
        
        candidate_scores = best_priority_scores.copy()
        temp = candidate_scores[i_idx]
        candidate_scores[i_idx] = candidate_scores[j_idx]
        candidate_scores[j_idx] = temp
        
        stations = _simulate_greedy(candidate_scores, times, predecessors, cycle_time)
        
        if stations < best_stations:
            best_stations = stations
            best_priority_scores = candidate_scores.copy()
            order = np.argsort(-best_priority_scores)
            
            total_time = np.sum(times)
            lower_bound = int(np.ceil(total_time / cycle_time))
            if best_stations == lower_bound:
                break
    
    return best_priority_scores
