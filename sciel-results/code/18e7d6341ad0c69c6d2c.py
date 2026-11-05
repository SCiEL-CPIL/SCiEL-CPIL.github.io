"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque

def _compute_heights(times, predecessors, n):
    """Compute longest path to sink (height) for each node."""
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            
    out_degree = [len(succs) for succs in successors]
    heights = np.zeros(n, dtype=float)
    
    from collections import deque
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
            candidate_height = times[p] + heights[u]
            if candidate_height > heights[p]:
                heights[p] = candidate_height
            out_degree[p] -= 1
            if out_degree[p] == 0:
                queue.append(p)
                
    if processed < n:
        for i in range(n):
            if heights[i] == 0 and times[i] > 0:
                heights[i] = times[i]
                
    return heights

def _simulate_greedy(priority_scores, times, predecessors, cycle_time):
    """
    Simulates the greedy assignment process using the given priority scores.
    Returns the number of stations used.
    """
    n = len(times)
    # Sort tasks by priority (highest first). 
    # Note: The evaluator likely uses a max-heap or sorts eligible tasks.
    # We mimic the standard "highest priority eligible task" logic.
    
    # Pre-compute successors for quick checking of eligibility? 
    # Actually, eligibility depends on which tasks are already assigned.
    # A task is eligible if all its predecessors are assigned.
    
    assigned = [False] * n
    station_load = 0.0
    num_stations = 1
    
    # We need to efficiently find the highest priority eligible task that fits.
    # Since N is small (30), we can just scan all tasks each time.
    # To make it faster, we can keep a list of unassigned tasks and filter.
    
    remaining_tasks = list(range(n))
    
    while remaining_tasks:
        best_task_idx = -1
        best_priority = -1.0
        
        # Find the highest priority task that is eligible and fits
        for i in remaining_tasks:
            if assigned[i]:
                continue
            
            # Check eligibility: all predecessors must be assigned
            eligible = True
            for p in predecessors[i]:
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
            remaining_tasks.remove(best_task_idx)
        else:
            # No task fits in the current station. Start a new station.
            # We must pick the highest priority eligible task, even if it doesn't fit?
            # Wait, the rule is "stations are filled with the highest-priority eligible task that fits the cycle time."
            # If NO eligible task fits, we start a new station.
            # In a new station, the constraint is just that the task's time <= cycle_time.
            # Since all tasks are generated to fit within cycle time usually, or we assume they do.
            # If a single task is > cycle_time, the problem is infeasible.
            # Assuming feasible instances.
            
            num_stations += 1
            station_load = 0.0
            
            # Now find the highest priority eligible task that fits in the NEW empty station.
            # This is the same logic as above, but station_load is 0.
            best_task_idx = -1
            best_priority = -1.0
            
            for i in remaining_tasks:
                if assigned[i]:
                    continue
                
                eligible = True
                for p in predecessors[i]:
                    if not assigned[p]:
                        eligible = False
                        break
                
                if not eligible:
                    continue
                
                # In a new station, any task with time <= cycle_time fits.
                if times[i] <= cycle_time:
                    if priority_scores[i] > best_priority:
                        best_priority = priority_scores[i]
                        best_task_idx = i
            
            if best_task_idx != -1:
                assigned[best_task_idx] = True
                station_load += times[best_task_idx]
                remaining_tasks.remove(best_task_idx)
            else:
                # This should not happen if the instance is feasible and we haven't assigned all.
                # It implies no eligible task exists, which is a contradiction if predecessors are satisfied.
                # Or it implies a cycle? Or we missed a task.
                # Break to avoid infinite loop.
                break
                
    return num_stations

def priority(instance):
    """
    Computes a priority score for each task by simulating the greedy balancing
    process with randomized perturbations to find a better-than-average static heuristic.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer before 150s limit
    
    # Base heuristic: Longest Path (Height)
    base_heights = _compute_heights(times, predecessors, n)
    
    # We will try to improve upon the base heuristic by adding small random noise
    # and simulating the result.
    # The evaluator uses the returned scores to pick the "highest priority eligible task that fits".
    # This is equivalent to sorting by priority.
    
    best_priority_scores = base_heights.copy()
    best_stations = _simulate_greedy(base_heights, times, predecessors, cycle_time)
    
    # Try a few variations
    # Variation 1: Pure Height
    # Variation 2: Height + small random noise
    # Variation 3: Height + Successor Count tie-breaker
    
    # Compute successor counts for tie-breaking
    successors_count = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            successors_count[p] += 1
    
    # Pre-compute a combined score for tie-breaking: Height * 1000 + SuccCount
    # This ensures height is primary, succ count is secondary.
    combined_base = base_heights * 1000.0 + successors_count
    
    # Simulate the combined base to see if it's better
    stations_combined = _simulate_greedy(combined_base, times, predecessors, cycle_time)
    if stations_combined < best_stations:
        best_stations = stations_combined
        best_priority_scores = combined_base.copy()
    
    # Now, do some randomized local search / perturbation
    # We want to find a priority vector that leads to fewer stations.
    # We can perturb the base heights slightly.
    
    # Set a random seed based on instance properties for reproducibility? 
    # No, the task says "five generated 30-task instances". It's better to be deterministic 
    # or use a fixed seed to ensure consistent performance across runs if the evaluator 
    # runs the same instance multiple times? 
    # The evaluator runs the function once per instance evaluation. 
    # Using a fixed seed ensures we always pick the same "good" perturbation if one exists.
    rng = random.Random(42)
    
    max_iterations = 100
    iteration = 0
    
    while time.time() < deadline and iteration < max_iterations:
        iteration += 1
        
        # Create a perturbed version of the base heights
        # Add small noise to break ties or slightly adjust ordering
        # The magnitude of noise should be small enough to not completely destroy the height logic,
        # but large enough to change the order of close tasks.
        
        # Strategy: Randomly swap priorities of two tasks that are close in height?
        # Or just add uniform noise?
        # Adding noise to all tasks might be too destructive.
        
        # Let's try: Take the combined_base, and add a small random vector.
        # Or better: Randomly select a subset of tasks and shuffle their relative order slightly?
        
        # A simpler approach: 
        # Generate a random vector of noise in [-0.1, 0.1]
        # Add it to the combined_base.
        # This effectively randomizes the tie-breaking and slightly perturbs the main order.
        
        noise = np.random.uniform(-0.5, 0.5, n)
        perturbed_scores = combined_base + noise
        
        stations = _simulate_greedy(perturbed_scores, times, predecessors, cycle_time)
        
        if stations < best_stations:
            best_stations = stations
            best_priority_scores = perturbed_scores.copy()
            
            # If we found a perfect solution (lower bound), we can stop?
            # Lower bound is ceil(total_time / cycle_time)
            total_time = np.sum(times)
            lower_bound = int(np.ceil(total_time / cycle_time))
            if best_stations == lower_bound:
                break

    return best_priority_scores
# EVOLVE-BLOCK-END
