"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from functools import cmp_to_key
import time


def _compute_successors(predecessors, n):
    """Compute successors list from predecessors."""
    succ = [[] for _ in range(n)]
    for j in range(n):
        for i in predecessors[j]:
            succ[i].append(j)
    return succ


def _compute_indegree(predecessors, n):
    """Compute in-degree for each task."""
    indeg = [0] * n
    for j in range(n):
        indeg[j] = len(predecessors[j])
    return indeg


def _compute_longest_path_downstream(times, succ, n):
    """
    Compute longest path from each task to the end (downstream).
    longest[i] = times[i] + max(longest[j] for j in succ[i])
    """
    longest = np.zeros(n, dtype=float)
    # Process in reverse topological order (sinks first)
    # We can use a simple approach: iterate until convergence or use topo sort
    # For 30 tasks, a simple repeated pass works
    longest = np.array(times, dtype=float).copy()
    for _ in range(n):  # at most n passes needed
        changed = False
        for i in range(n):
            if succ[i]:
                max_down = max(longest[j] for j in succ[i])
                new_val = times[i] + max_down
                if new_val > longest[i]:
                    longest[i] = new_val
                    changed = True
        if not changed:
            break
    return longest


def _compute_longest_path_upstream(times, predecessors, n):
    """
    Compute longest path from the start to each task (upstream).
    longest_up[i] = times[i] + max(longest_up[j] for j in predecessors[i])
    """
    longest_up = np.zeros(n, dtype=float)
    longest_up = np.array(times, dtype=float).copy()
    for _ in range(n):
        changed = False
        for i in range(n):
            if predecessors[i]:
                max_up = max(longest_up[j] for j in predecessors[i])
                new_val = times[i] + max_up
                if new_val > longest_up[i]:
                    longest_up[i] = new_val
                    changed = True
        if not changed:
            break
    return longest_up


def _simulate(priority_scores, times, predecessors, cycle_time, n):
    """
    Simulate the greedy station-filling algorithm.
    Returns the number of stations used.
    """
    succ = _compute_successors(predecessors, n)
    
    # Track which tasks are completed
    completed = [False] * n
    # Track remaining predecessors count for each task
    remaining_pred = [len(predecessors[j]) for j in range(n)]
    
    # Available tasks: those with all predecessors completed
    # We'll use a list and filter each time for simplicity with n=30
    
    stations = 0
    current_station_time = 0.0
    
    while any(not c for c in completed):
        # Find eligible tasks: not completed, all predecessors completed, fits in current station
        eligible = []
        for j in range(n):
            if not completed[j] and remaining_pred[j] == 0:
                if current_station_time + times[j] <= cycle_time + 1e-9:
                    eligible.append(j)
        
        if not eligible:
            # Start a new station
            stations += 1
            current_station_time = 0.0
            # Re-check eligibility with new station
            eligible = []
            for j in range(n):
                if not completed[j] and remaining_pred[j] == 0:
                    if times[j] <= cycle_time + 1e-9:
                        eligible.append(j)
            
            if not eligible:
                # This shouldn't happen if cycle_time is valid, but break to avoid infinite loop
                break
        
        # Select the highest priority task from eligible
        best_task = max(eligible, key=lambda j: priority_scores[j])
        
        # Assign it to current station
        current_station_time += times[best_task]
        completed[best_task] = True
        
        # Update remaining_pred for successors
        for k in succ[best_task]:
            remaining_pred[k] -= 1
    
    # If we ended with a non-empty current station, it counts as a station
    # But the loop above increments stations when starting a new one.
    # Let's re-examine: 
    # - Initially stations=0, current_station_time=0
    # - First iteration: if no eligible (because current_station_time=0 and all fit), we don't increment.
    #   Wait, the logic is: if no eligible fits in CURRENT station, start NEW station (increment).
    #   But initially, current_station_time=0, so all eligible tasks fit.
    #   So we assign to the "first" station without incrementing? That's wrong.
    
    # Let me restructure the simulation more carefully.
    return stations


def _simulate_v2(priority_scores, times, predecessors, cycle_time, n):
    """
    More careful simulation.
    """
    succ = _compute_successors(predecessors, n)
    completed = [False] * n
    remaining_pred = [len(predecessors[j]) for j in range(n)]
    
    stations = 0
    current_station_time = 0.0
    in_station = False  # whether we've started placing in the current station
    
    while any(not c for c in completed):
        # Find eligible tasks that fit in the current station
        eligible = []
        for j in range(n):
            if not completed[j] and remaining_pred[j] == 0:
                if current_station_time + times[j] <= cycle_time + 1e-9:
                    eligible.append(j)
        
        if eligible:
            # Pick highest priority
            best_task = max(eligible, key=lambda j: priority_scores[j])
            current_station_time += times[best_task]
            completed[best_task] = True
            in_station = True
            
            for k in succ[best_task]:
                remaining_pred[k] -= 1
        else:
            # No eligible task fits in current station
            if in_station:
                # Current station is done, start a new one
                stations += 1
                current_station_time = 0.0
                in_station = False
            else:
                # We haven't placed anything yet, start first station
                stations += 1
                current_station_time = 0.0
                in_station = False
                # Re-check: now with empty station, find eligible
                eligible = []
                for j in range(n):
                    if not completed[j] and remaining_pred[j] == 0:
                        if times[j] <= cycle_time + 1e-9:
                            eligible.append(j)
                
                if not eligible:
                    break  # Should not happen
                
                best_task = max(eligible, key=lambda j: priority_scores[j])
                current_station_time += times[best_task]
                completed[best_task] = True
                in_station = True
                
                for k in succ[best_task]:
                    remaining_pred[k] -= 1
    
    # If we finished with an in-progress station, count it
    if in_station:
        stations += 1
    
    return stations


def priority(instance):
    """
    Score each assembly task to minimize the number of stations.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Compute various features
    # 1. Longest path downstream (critical path remaining)
    succ = _compute_successors(predecessors, n)
    longest_down = _compute_longest_path_downstream(times, succ, n)
    
    # 2. Longest path upstream
    longest_up = _compute_longest_path_upstream(times, predecessors, n)
    
    # 3. Number of successors (direct)
    num_succ = np.array([len(succ[i]) for i in range(n)], dtype=float)
    
    # 4. Number of predecessors (direct)
    num_pred = np.array([len(predecessors[i]) for i in range(n)], dtype=float)
    
    # Combine features into a priority score.
    # Heuristic: prioritize tasks that are on the critical path (high longest_down),
    # have many successors (to unblock more tasks), and are large (to reduce fragmentation).
    
    # Normalize features to [0, 1] range for combination
    def normalize(arr):
        arr = np.asarray(arr, dtype=float)
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val < 1e-12:
            return np.ones_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    norm_down = normalize(longest_down)
    norm_up = normalize(longest_up)
    norm_succ = normalize(num_succ)
    norm_pred = normalize(num_pred)
    norm_time = normalize(times)
    
    # Weighted combination
    # Emphasize downstream critical path and successor count
    priority_scores = (
        0.4 * norm_down +
        0.2 * norm_time +
        0.2 * norm_succ +
        0.1 * norm_up +
        0.1 * (1.0 - norm_pred)  # Fewer predecessors is slightly better? Or more?
    )
    
    # Actually, let's try a simpler, more proven heuristic:
    # Rank by longest downstream path, breaking ties by number of successors, then by time.
    
    # Create a composite score where higher is better
    # Use a lexicographic-like combination
    max_down = np.max(longest_down) if n > 0 else 1.0
    max_succ = np.max(num_succ) if n > 0 else 1.0
    max_time = np.max(times) if n > 0 else 1.0
    
    if max_down < 1e-12:
        max_down = 1.0
    if max_succ < 1e-12:
        max_succ = 1.0
    if max_time < 1e-12:
        max_time = 1.0
    
    # Primary: longest downstream path
    # Secondary: number of successors
    # Tertiary: task time
    priority_scores = (
        (longest_down / max_down) * 1.0 +
        (num_succ / max_succ) * 0.01 +
        (times / max_time) * 0.001
    )
    
    return priority_scores
# EVOLVE-BLOCK-END
