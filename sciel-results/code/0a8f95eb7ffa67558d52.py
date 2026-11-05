"""
Strategy: Improved Greedy with Better Initial Heuristics and Faster Local Search

The key improvements over the previous approach:
1. Use longest path (critical path) as the primary heuristic, which is the standard
   and often optimal priority rule for assembly line balancing.
2. Faster simulation using a more efficient data structure.
3. More effective local search that focuses on improving the station count directly.
4. Better tie-breaking to avoid degenerate cases.
"""
import numpy as np
import time
from collections import deque

def _compute_longest_path(times, predecessors, n):
    """
    Compute the longest path from each task to any sink (including the task itself).
    This is the standard 'critical path' or 'rank' heuristic.
    """
    # Build successor list
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
    
    # Longest path in reverse topological order
    longest_path = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        longest_path[u] = times[u]
        for v in successors[u]:
            if longest_path[u] < times[u] + longest_path[v]:
                longest_path[u] = times[u] + longest_path[v]
    
    return longest_path, successors

def _simulate_greedy_fast(priority_scores, times, predecessors, cycle_time):
    """
    Fast simulation of greedy assignment.
    Returns number of stations.
    """
    n = len(times)
    t_list = list(times)
    p_list = list(priority_scores)
    
    assigned = [False] * n
    # Precompute predecessor sets as bitmasks for fast eligibility check
    pred_masks = [0] * n
    for i in range(n):
        mask = 0
        for p in predecessors[i]:
            mask |= (1 << p)
        pred_masks[i] = mask
    
    assigned_mask = 0
    station_load = 0.0
    num_stations = 1
    remaining = n
    
    while remaining > 0:
        best_idx = -1
        best_prio = -np.inf
        
        # Find best eligible task that fits
        for i in range(n):
            if (assigned_mask >> i) & 1:
                continue
            if (assigned_mask & pred_masks[i]) != pred_masks[i]:
                continue
            if station_load + t_list[i] <= cycle_time:
                if p_list[i] > best_prio:
                    best_prio = p_list[i]
                    best_idx = i
        
        if best_idx != -1:
            assigned_mask |= (1 << best_idx)
            station_load += t_list[best_idx]
            remaining -= 1
        else:
            num_stations += 1
            station_load = 0.0
            # Find best eligible task for new station
            for i in range(n):
                if (assigned_mask >> i) & 1:
                    continue
                if (assigned_mask & pred_masks[i]) != pred_masks[i]:
                    continue
                if t_list[i] <= cycle_time:
                    if p_list[i] > best_prio:
                        best_prio = p_list[i]
                        best_idx = i
            
            if best_idx != -1:
                assigned_mask |= (1 << best_idx)
                station_load += t_list[best_idx]
                remaining -= 1
            else:
                break
    
    return num_stations

def _simulate_greedy_with_stations(priority_scores, times, predecessors, cycle_time):
    """
    Simulation that also returns the station assignments for analysis.
    Returns (num_stations, station_assignments) where station_assignments[i] = station index.
    """
    n = len(times)
    t_list = list(times)
    p_list = list(priority_scores)
    
    pred_masks = [0] * n
    for i in range(n):
        mask = 0
        for p in predecessors[i]:
            mask |= (1 << p)
        pred_masks[i] = mask
    
    assigned_mask = 0
    station_load = 0.0
    num_stations = 1
    remaining = n
    station_assignments = [-1] * n
    current_station = 0
    
    while remaining > 0:
        best_idx = -1
        best_prio = -np.inf
        
        for i in range(n):
            if (assigned_mask >> i) & 1:
                continue
            if (assigned_mask & pred_masks[i]) != pred_masks[i]:
                continue
            if station_load + t_list[i] <= cycle_time:
                if p_list[i] > best_prio:
                    best_prio = p_list[i]
                    best_idx = i
        
        if best_idx != -1:
            assigned_mask |= (1 << best_idx)
            station_load += t_list[best_idx]
            station_assignments[best_idx] = current_station
            remaining -= 1
        else:
            num_stations += 1
            current_station += 1
            station_load = 0.0
            for i in range(n):
                if (assigned_mask >> i) & 1:
                    continue
                if (assigned_mask & pred_masks[i]) != pred_masks[i]:
                    continue
                if t_list[i] <= cycle_time:
                    if p_list[i] > best_prio:
                        best_prio = p_list[i]
                        best_idx = i
            
            if best_idx != -1:
                assigned_mask |= (1 << best_idx)
                station_load += t_list[best_idx]
                station_assignments[best_idx] = current_station
                remaining -= 1
            else:
                break
    
    return num_stations, station_assignments

def priority(instance):
    """
    Computes priority scores using longest path heuristic with local search refinement.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Compute longest path (critical path)
    longest_path, successors = _compute_longest_path(times, predecessors, n)
    
    # 2. Compute additional features
    # Number of immediate successors
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)
    # Number of immediate predecessors
    pred_count = np.array([len(predecessors[i]) for i in range(n)], dtype=float)
    # Total downstream work (longest path already captures this)
    
    # Normalize features
    def normalize(arr):
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-12:
            return np.zeros(len(arr))
        return (arr - mn) / (mx - mn)
    
    lp_norm = normalize(longest_path)
    sc_norm = normalize(succ_count)
    pc_norm = normalize(pred_count)
    time_norm = normalize(times)
    
    # 3. Try several priority combinations
    rng = np.random.default_rng(123)
    
    # Weight combinations: (longest_path, succ_count, pred_count, time)
    weight_combos = [
        (1.0, 0.0, 0.0, 0.0),    # Pure longest path
        (0.8, 0.2, 0.0, 0.0),    # LP + succ
        (0.7, 0.1, 0.1, 0.1),    # LP + small others
        (0.9, 0.0, 0.0, 0.1),    # LP + time
        (0.6, 0.2, 0.1, 0.1),    # More balanced
    ]
    
    best_scores = None
    best_stations = float('inf')
    
    total_time = np.sum(times)
    lower_bound = int(np.ceil(total_time / cycle_time))
    
    for w_lp, w_sc, w_pc, w_t in weight_combos:
        if time.time() > deadline - 60.0:
            break
        
        candidate = w_lp * lp_norm + w_sc * sc_norm + w_pc * pc_norm + w_t * time_norm
        # Add tiny tie-breaker
        candidate = candidate + np.arange(n) * 1e-10
        
        stations = _simulate_greedy_fast(candidate, times, predecessors, cycle_time)
        
        if stations < best_stations:
            best_stations = stations
            best_scores = candidate.copy()
            if best_stations == lower_bound:
                break
    
    if best_scores is None:
        best_scores = longest_path.copy() + np.arange(n) * 1e-10
        best_stations = _simulate_greedy_fast(best_scores, times, predecessors, cycle_time)
    
    # 4. Local search: pairwise swaps
    # Sort by priority to get rank order
    order = np.argsort(-best_scores)
    
    n_improvements = 0
    max_swaps = 5000
    
    while time.time() < deadline - 10.0 and n_improvements < max_swaps:
        # Pick two tasks to swap
        # Bias toward swapping tasks that are close in rank
        i_idx = int(rng.integers(0, n))
        rank_i = int(np.where(order == i_idx)[0][0])
        
        # Choose offset
        if n_improvements < 100:
            # Early: small offsets
            rank_offset = int(rng.integers(1, 4))
        else:
            # Later: larger offsets
            rank_offset = int(rng.integers(1, 10))
        
        if rank_i - rank_offset >= 0:
            rank_j = rank_i - rank_offset
        elif rank_i + rank_offset < n:
            rank_j = rank_i + rank_offset
        else:
            continue
        
        j_idx = int(order[rank_j])
        if i_idx == j_idx:
            continue
        
        # Swap priorities
        candidate = best_scores.copy()
        temp = candidate[i_idx]
        candidate[i_idx] = candidate[j_idx]
        candidate[j_idx] = temp
        
        stations = _simulate_greedy_fast(candidate, times, predecessors, cycle_time)
        
        if stations < best_stations:
            best_stations = stations
            best_scores = candidate
            order = np.argsort(-best_scores)
            n_improvements = 0  # Reset counter on improvement
            if best_stations == lower_bound:
                break
        else:
            n_improvements += 1
    
    # 5. Final polish: try all pairs of tasks within a small rank window
    if time.time() < deadline - 2.0 and best_stations > lower_bound:
        order = np.argsort(-best_scores)
        window = 5
        found_improvement = True
        
        while found_improvement and time.time() < deadline - 1.0:
            found_improvement = False
            for rank_i in range(n):
                if time.time() > deadline - 1.0:
                    break
                for offset in range(1, min(window, n - rank_i)):
                    rank_j = rank_i + offset
                    i_idx = int(order[rank_i])
                    j_idx = int(order[rank_j])
                    
                    candidate = best_scores.copy()
                    temp = candidate[i_idx]
                    candidate[i_idx] = candidate[j_idx]
                    candidate[j_idx] = temp
                    
                    stations = _simulate_greedy_fast(candidate, times, predecessors, cycle_time)
                    
                    if stations < best_stations:
                        best_stations = stations
                        best_scores = candidate
                        order = np.argsort(-best_scores)
                        found_improvement = True
                        if best_stations == lower_bound:
                            break
                if best_stations == lower_bound:
                    break
    
    return best_scores
