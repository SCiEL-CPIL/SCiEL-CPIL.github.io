"""
Strategy: Local Search (Hill Climbing) on Priority Vector
Instead of a fixed heuristic, we treat the priority scores as variables to be optimized.
We start with a good heuristic (Longest Path) and then iteratively perturb the priorities
(swap two tasks' priorities) and evaluate the resulting number of stations.
If the perturbation improves or equals the station count (with some probability for diversity),
we keep it. This adapts the priority rule to the specific instance structure.
"""

import numpy as np
import time
import random
from collections import deque

def _simulate_greedy(times, predecessors, cycle_time, priorities):
    """
    Simulates the greedy line balancing process.
    Returns the number of stations used.
    """
    n = len(times)
    if n == 0:
        return 0
    
    # Build successor list and in-degrees for fast eligibility check
    # predecessors[i] is list of j such that j -> i
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    # Initial eligible set: tasks with in_degree 0
    eligible = set(i for i in range(n) if in_degree[i] == 0)
    
    # Current station load
    current_load = 0.0
    stations = 0
    
    # We need to pick the highest priority eligible task that fits.
    # Since n is small (30), we can just scan eligible set each time.
    # To make it faster, we could use a heap, but n is small.
    
    remaining = set(range(n))
    
    while remaining:
        # Find best task to add to current station
        best_task = -1
        best_priority = -np.inf
        best_time = 0.0
        
        for t in eligible:
            if t not in remaining:
                continue
            if times[t] <= cycle_time - current_load:
                if priorities[t] > best_priority:
                    best_priority = priorities[t]
                    best_task = t
                    best_time = times[t]
                    
        if best_task == -1:
            # No task fits in current station. Move to next station.
            if current_load > 0:
                # If current station was empty, we have a problem (cycle or invalid data)
                # But assuming valid DAG, this shouldn't happen if eligible is not empty
                # If eligible is empty but remaining is not, it's a cycle.
                # For safety, break to avoid infinite loop.
                if not eligible:
                    break
                stations += 1
                current_load = 0.0
                # If eligible is empty, we might need to break or handle cycle.
                # In a DAG, if remaining is not empty, eligible must not be empty.
                continue
            else:
                # Current station is empty and no task fits?
                # This implies times[t] > cycle_time for all t in eligible.
                # This is an infeasible instance.
                return float('inf')
            
        # Assign best_task to current station
        current_load += best_time
        remaining.remove(best_task)
        eligible.discard(best_task)
        
        # Update eligibility of successors
        for s in successors[best_task]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                eligible.add(s)
                
    # Don't forget the last station
    if current_load > 0:
        stations += 1
        
    return stations

def _compute_heights(times, predecessors):
    """
    Computes longest path to sink (height) for initial priority.
    """
    n = len(times)
    successors = [[] for _ in range(n)]
    out_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            out_degree[p] += 1
            
    heights = np.zeros(n, dtype=float)
    queue = deque()
    for i in range(n):
        if out_degree[i] == 0:
            queue.append(i)
            heights[i] = times[i]
            
    processed = 0
    while queue:
        u = queue.popleft()
        processed += 1
        for p in predecessors[u]:
            candidate = times[p] + heights[u]
            if candidate > heights[p]:
                heights[p] = candidate
            out_degree[p] -= 1
            if out_degree[p] == 0:
                queue.append(p)
                
    if processed < n:
        for i in range(n):
            if heights[i] == 0:
                heights[i] = times[i]
                
    return heights

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # 1. Initial Solution: Longest Path (Height)
    current_priorities = _compute_heights(times, predecessors)
    current_stations = _simulate_greedy(times, predecessors, cycle_time, current_priorities)
    
    if n == 0:
        return current_priorities
        
    # 2. Local Search
    # Deadline guard
    start_time = time.time()
    timeout = 120.0  # Safe margin before 150s
    
    # Temperature for simulated annealing-like behavior
    # We'll use a simple hill climbing with random restarts or swaps.
    # Given the small size, pure hill climbing with random swaps is efficient.
    
    # To avoid getting stuck in local minima, we can allow "neutral" moves
    # (same station count) with a certain probability, or use a slight noise.
    
    best_priorities = current_priorities.copy()
    best_stations = current_stations
    
    # Pre-compute some stats? Not necessary for n=30.
    
    iterations = 0
    while time.time() - start_time < timeout:
        iterations += 1
        
        # Perturbation: Swap priorities of two random tasks
        i, j = random.sample(range(n), 2)
        
        # Save old priorities
        old_i, old_j = current_priorities[i], current_priorities[j]
        
        # Swap
        current_priorities[i], current_priorities[j] = old_j, old_i
        
        # Evaluate
        new_stations = _simulate_greedy(times, predecessors, cycle_time, current_priorities)
        
        # Acceptance Criterion
        if new_stations < current_stations:
            # Improvement
            current_stations = new_stations
            if new_stations < best_stations:
                best_stations = new_stations
                best_priorities = current_priorities.copy()
        elif new_stations == current_stations:
            # Neutral move: accept with some probability to explore plateaus
            if random.random() < 0.5:
                current_stations = new_stations
            else:
                # Revert
                current_priorities[i], current_priorities[j] = old_i, old_j
        else:
            # Worse: revert
            current_priorities[i], current_priorities[j] = old_i, old_j
            
        # Check time frequently to ensure we break out quickly
        if time.time() - start_time > timeout - 1.0:
            break
            
    return best_priorities
