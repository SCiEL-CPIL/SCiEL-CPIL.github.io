import numpy as np
import time
from collections import defaultdict, deque
import heapq

def _greedy_balance(times, predecessors, successors, n, cycle_time, priority_scores):
    """Simulate greedy station filling given a priority vector. Returns (num_stations, task_to_station)."""
    task_to_station = [-1] * n
    remaining_preds = [len(predecessors[i]) if predecessors[i] else 0 for i in range(n)]
    remaining_succs = [len(successors[i]) for i in range(n)]
    
    # Eligible: all predecessors assigned
    # Use a max-heap by priority (negate for min-heap)
    eligible = []
    for i in range(n):
        if remaining_preds[i] == 0:
            heapq.heappush(eligible, (-priority_scores[i], i))
    
    num_stations = 0
    current_station_load = 0.0
    current_station_tasks = []
    
    while eligible:
        # Get highest priority eligible task
        neg_pri, task = heapq.heappop(eligible)
        
        # Check if it fits in current station
        if current_station_load + times[task] <= cycle_time + 1e-9:
            # Assign to current station
            current_station_load += times[task]
            current_station_tasks.append(task)
            task_to_station[task] = num_stations
            
            # Update successors
            for succ in successors[task]:
                remaining_preds[succ] -= 1
                if remaining_preds[succ] == 0:
                    heapq.heappush(eligible, (-priority_scores[succ], succ))
            
            # If station is full (can't fit any more), close it
            # Actually, we should try to fill it optimally, but greedy just keeps adding
            # Let's check if we should close: if no eligible task fits, close
            # For simplicity in simulation, just keep adding if it fits
        else:
            # Close current station, open new one
            num_stations += 1
            current_station_load = times[task]
            current_station_tasks = [task]
            task_to_station[task] = num_stations
            
            # Put this task back for the new station (it was already popped)
            # Actually it's already assigned, so just continue
            # But we need to re-try: the task didn't fit in old station, 
            # so it starts the new station. No need to push back.
            
            # Update successors
            for succ in successors[task]:
                remaining_preds[succ] -= 1
                if remaining_preds[succ] == 0:
                    heapq.heappush(eligible, (-priority_scores[succ], succ))
    
    # Count total stations (0-indexed, so num_stations + 1 if any tasks were assigned)
    if n > 0:
        max_station = max(task_to_station) if any(t >= 0 for t in task_to_station) else 0
        num_stations = max_station + 1
    else:
        num_stations = 0
    
    return num_stations, task_to_station


def _compute_station_utilization(times, task_to_station, n, cycle_time):
    """Compute how well each station is packed. Returns array of utilization per task's station."""
    if n == 0:
        return np.zeros(0)
    
    max_station = max(task_to_station) if any(t >= 0 for t in task_to_station) else 0
    num_stations = max_station + 1
    
    station_load = np.zeros(num_stations)
    for i in range(n):
        if task_to_station[i] >= 0:
            station_load[task_to_station[i]] += times[i]
    
    # Utilization of each station
    station_util = station_load / max(cycle_time, 1e-6)
    
    # For each task, its station's utilization
    task_util = np.zeros(n)
    for i in range(n):
        if task_to_station[i] >= 0:
            task_util[i] = station_util[task_to_station[i]]
    
    return task_util


def priority(instance):
    start_time = time.time()
    deadline = start_time + 8.0  # safety margin well under 120s
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    if n == 0:
        return np.array([], dtype=float)
    
    cycle_time = instance.get("cycle_time", 40.0)
    
    # Build adjacency structures
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        if predecessors[i]:
            for pred in predecessors[i]:
                successors[pred].append(i)
                in_degree[i] += 1
    
    # Topological order using Kahn's algorithm
    topo_order = []
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    temp_in_degree = in_degree[:]
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            temp_in_degree[v] -= 1
            if temp_in_degree[v] == 0:
                queue.append(v)
    
    if len(topo_order) < n:
        for i in range(n):
            if i not in set(topo_order):
                topo_order.append(i)
    
    # Compute Total Downstream Work (TDW)
    tdw = [0.0] * n
    for i in reversed(topo_order):
        tdw[i] = times[i] + sum(tdw[j] for j in successors[i])
    
    # Compute Longest Successor Path (LSP)
    lsp = [0.0] * n
    for i in reversed(topo_order):
        max_succ_lsp = 0.0
        for j in successors[i]:
            if lsp[j] > max_succ_lsp:
                max_succ_lsp = lsp[j]
        lsp[i] = times[i] + max_succ_lsp
    
    # Compute Longest Predecessor Path (LPP)
    lpp = [0.0] * n
    for i in topo_order:
        max_pred_lpp = 0.0
        if predecessors[i]:
            for pred in predecessors[i]:
                if lpp[pred] > max_pred_lpp:
                    max_pred_lpp = lpp[pred]
        lpp[i] = max_pred_lpp + times[i]
    
    # Transitive successor count
    trans_succ_count = [0] * n
    for i in reversed(topo_order):
        trans_succ_count[i] = len(successors[i]) + sum(trans_succ_count[j] for j in successors[i])
    trans_succ_count = np.array(trans_succ_count, dtype=float)
    
    # Transitive predecessor count
    trans_pred_count = [0] * n
    for i in topo_order:
        trans_pred_count[i] = len(predecessors[i]) if predecessors[i] else 0
        for p in predecessors[i]:
            trans_pred_count[i] += trans_pred_count[p]
    trans_pred_count = np.array(trans_pred_count, dtype=float)
    
    # Base priority from structural heuristics
    tdw_arr = np.asarray(tdw, dtype=float)
    lsp_arr = np.asarray(lsp, dtype=float)
    lpp_arr = np.asarray(lpp, dtype=float)
    
    tdw_max = np.max(tdw_arr) if np.max(tdw_arr) > 0 else 1.0
    lsp_max = np.max(lsp_arr) if np.max(lsp_arr) > 0 else 1.0
    lpp_max = np.max(lpp_arr) if np.max(lpp_arr) > 0 else 1.0
    trans_succ_max = np.max(trans_succ_count) if np.max(trans_succ_count) > 0 else 1.0
    trans_pred_max = np.max(trans_pred_count) if np.max(trans_pred_count) > 0 else 1.0
    
    base_priority = (
        1.0 * (tdw_arr / tdw_max) +
        0.5 * (lsp_arr / lsp_max) +
        0.3 * (lpp_arr / lpp_max) +
        0.2 * (trans_succ_count / trans_succ_max) +
        0.1 * (trans_pred_count / trans_pred_max) +
        times * 1e-6
    )
    
    # Now do simulated annealing / iterative refinement:
    # 1. Simulate with base priority, get station assignments
    # 2. Compute "station packing quality" per task
    # 3. Adjust priorities based on which tasks ended up in poorly-packed stations
    # 4. Repeat for a few iterations
    
    best_priority = base_priority.copy()
    best_num_stations = float('inf')
    
    num_iterations = 5
    for it in range(num_iterations):
        if time.time() > deadline - 2.0:
            break
        
        # Simulate with current priority
        num_stations, task_to_station = _greedy_balance(
            times, predecessors, successors, n, cycle_time, best_priority
        )
        
        if num_stations < best_num_stations:
            best_num_stations = num_stations
        
        # Compute station utilization
        station_util = _compute_station_utilization(times, task_to_station, n, cycle_time)
        
        # Tasks in poorly-utilized stations should get higher priority 
        # (they were "wasted" capacity; pulling them earlier might help)
        # Actually, tasks in well-packed stations are "good" - keep their relative order
        # Tasks in poorly-packed stations: if they're large, they might be better scheduled differently
        
        # New approach: boost priority of tasks that are in stations with high utilization
        # (they contribute to good packing) and slightly penalize those in low-utilization stations
        # This reinforces good packing patterns
        
        util_mean = np.mean(station_util) if n > 0 else 0
        util_std = np.std(station_util) if n > 0 else 1
        if util_std < 1e-6:
            util_std = 1e-6
        
        # Normalize utilization to [0, 1] range
        util_norm = (station_util - np.min(station_util)) / (np.max(station_util) - np.min(station_util) + 1e-6)
        
        # Blend: keep base structural priority but add a component that rewards
        # being in well-packed stations (reinforcement learning style)
        blend_weight = 0.3 * (it + 1) / num_iterations  # gradually increase reinforcement
        reinforced_priority = (1 - blend_weight) * base_priority + blend_weight * (
            0.5 * base_priority + 0.5 * util_norm
        )
        
        # Also add a small random perturbation for diversity
        rng = np.random.RandomState(42 + it)
        perturbation = rng.randn(n) * 0.01
        reinforced_priority = reinforced_priority + perturbation
        
        # Ensure priorities are positive
        reinforced_priority = np.maximum(reinforced_priority, 1e-6)
        
        # Test this new priority
        test_num_stations, _ = _greedy_balance(
            times, predecessors, successors, n, cycle_time, reinforced_priority
        )
        
        if test_num_stations <= best_num_stations:
            best_priority = reinforced_priority.copy()
            if test_num_stations < best_num_stations:
                best_num_stations = test_num_stations
        else:
            # Revert: keep best_priority as is
            pass
    
    # Final: return the best priority found
    # Add a tiny time-based tiebreaker to ensure determinism
    final_priority = best_priority + times * 1e-8
    
    return final_priority
