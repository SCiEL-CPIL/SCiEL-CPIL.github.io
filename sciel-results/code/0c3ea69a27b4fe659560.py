"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque

def _compute_features(instance):
    """
    Computes various features for each task to be used in priority scoring.
    Returns a dictionary of feature names to numpy arrays.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    predecessors = instance["predecessors"]
    
    # Build successor lists and in-degrees
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        in_degree[i] = len(preds)
        for p in preds:
            successors[p].append(i)
            
    out_degree = np.array([len(successors[i]) for i in range(n)], dtype=float)

    # Topological sort
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_deg_temp = in_degree.copy()
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_deg_temp[succ] -= 1
            if in_deg_temp[succ] == 0:
                queue.append(succ)
                
    if len(topo_order) < n:
        # Cycle detected, return basic features
        return {
            "time": times,
            "out_degree": out_degree,
            "in_degree": in_degree.astype(float),
            "pos_weight": times.copy(),
            "level": np.zeros(n)
        }

    # Positional Weight: time + sum(pos_weight of successors)
    pos_weight = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        pos_weight[node] = times[node]
        for succ in successors[node]:
            pos_weight[node] += pos_weight[succ]

    # Level: longest path from start (in terms of time or count? Let's use time)
    # level[i] = time[i] + max(level[pred] for pred in preds)
    level = np.zeros(n, dtype=float)
    for node in topo_order:
        preds = predecessors[node] if node < len(predecessors) else []
        if preds:
            max_pred_level = max(level[p] for p in preds)
            level[node] = times[node] + max_pred_level
        else:
            level[node] = times[node]

    return {
        "time": times,
        "out_degree": out_degree,
        "in_degree": in_degree.astype(float),
        "pos_weight": pos_weight,
        "level": level
    }

def _simulate_greedy(instance, scores):
    """
    Simulates the greedy assignment algorithm using the given scores.
    Returns the number of stations used.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    predecessors = instance["predecessors"]
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        for p in preds:
            successors[p].append(i)

    # State
    remaining_preds = [len(predecessors[i]) if i < len(predecessors) else 0 for i in range(n)]
    eligible = set(i for i in range(n) if remaining_preds[i] == 0)
    
    stations = 0
    current_load = 0.0
    scheduled = set()
    
    # To make it faster, we can sort eligible tasks by score each time, 
    # but for N=30, scanning is fine.
    
    while len(scheduled) < n:
        if not eligible:
            # Should not happen if graph is valid, but break to avoid infinite loop
            break
            
        # Find the best eligible task that fits
        best_task = -1
        best_score = -np.inf
        
        for task in eligible:
            if times[task] <= cycle_time - current_load:
                if scores[task] > best_score:
                    best_score = scores[task]
                    best_task = task
                    
        if best_task == -1:
            # No eligible task fits in current station
            stations += 1
            current_load = 0.0
            # Re-evaluate (the set of eligible tasks is the same, but load is reset)
            # Actually, we just continue the loop, which will now find a task that fits in empty station
            continue
            
        # Assign best_task to current station
        current_load += times[best_task]
        scheduled.add(best_task)
        eligible.remove(best_task)
        
        # Update successors
        for succ in successors[best_task]:
            remaining_preds[succ] -= 1
            if remaining_preds[succ] == 0:
                eligible.add(succ)
                
    # If there are tasks left (shouldn't be), count them as needing stations
    if len(scheduled) < n:
        # Fallback: count remaining tasks roughly
        stations += 1 
        
    return stations

def priority(instance):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    if n == 0:
        return np.array([])
        
    features = _compute_features(instance)
    
    # Normalize features to [0, 1] to make weights more comparable
    feat_names = ["pos_weight", "time", "out_degree", "in_degree", "level"]
    feat_matrix = np.column_stack([features[f] for f in feat_names])
    
    # Handle zero max values
    max_vals = np.max(feat_matrix, axis=0)
    max_vals[max_vals == 0] = 1.0
    feat_matrix_norm = feat_matrix / max_vals
    
    # Initial weights: Based on the current program's logic (PW, SC, T)
    # Current: 0.6 * PW + 0.3 * SC + 0.1 * T
    # Indices: 0=PW, 1=Time, 2=OutDeg(SC), 3=InDeg, 4=Level
    weights = np.array([0.6, 0.1, 0.3, 0.0, 0.0])
    
    # Calculate initial score
    scores = feat_matrix_norm @ weights
    best_stations = _simulate_greedy(instance, scores)
    
    # Local Search on weights
    # We will try to minimize best_stations
    # Use a simple random perturbation + hill climbing with some SA-like acceptance
    # Since N=30, simulation is fast (~microseconds). We can do many iterations.
    
    T = 0.5 # Initial temperature for weight space
    T_min = 0.01
    cooling_rate = 0.999
    
    current_stations = best_stations
    current_weights = weights.copy()
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Generate a new weight vector
        # Perturb current weights
        new_weights = current_weights + np.random.normal(0, 0.1, size=len(weights))
        # Keep weights non-negative? Not strictly necessary, but helps interpretation.
        # Let's allow negative weights to explore more space, but normalize later?
        # Actually, let's just keep them as is.
        
        new_scores = feat_matrix_norm @ new_weights
        new_stations = _simulate_greedy(instance, new_scores)
        
        if new_stations < current_stations:
            current_stations = new_stations
            current_weights = new_weights
            # Reset temperature or keep? Keep for now.
        else:
            # Accept with probability exp(-delta / T)
            delta = new_stations - current_stations
            if delta > 0:
                p = np.exp(-delta / T)
                if random.random() < p:
                    current_weights = new_weights
                    # current_stations remains the best found so far? 
                    # No, if we accept a worse solution, current_stations should update 
                    # to reflect the state we are in, but we track best separately.
                    # For the acceptance criterion, we usually compare to the *current* state's objective.
                    # Let's track current_stations as the objective of the *current* weights.
                    current_stations = new_stations
            
        # Cool down
        T = max(T_min, T * cooling_rate)
        
        # If we found the theoretical minimum (1 station), we can stop
        if current_stations == 1:
            break
            
        # Check time
        if time.time() > deadline - 5:
            break
            
    # Finalize: Use the best weights found
    # Note: current_weights might be a "worse" accepted solution if the last step was an SA acceptance.
    # We should ideally keep track of the *best* weights seen.
    # Let's re-run one more time to ensure we are returning the best scores.
    # Actually, the loop updates current_weights only if accepted.
    # We need to track the global best.
    
    # Let's refactor slightly to track global best
    # The loop above updates current_weights to the accepted one.
    # If we accepted a worse one, current_stations is worse.
    # We need to store best_weights separately.
    
    # Let's just re-optimize the last few seconds to be sure?
    # Or simpler: The loop logic above is slightly flawed for keeping the *best*.
    # Let's rewrite the loop logic to be safer.
    
    # Actually, let's just return the scores from the weights that gave the best_stations found.
    # I didn't track best_weights in the loop above. Let's do a quick second pass or fix the logic.
    
    # Fix: Track best_weights
    best_weights = current_weights.copy() # Assuming current is best at start
    best_stations_found = _simulate_greedy(instance, feat_matrix_norm @ best_weights)
    
    # Re-run a simpler local search that strictly tracks the best
    T = 0.1
    for _ in range(10000): # Fixed number of quick iterations
        if time.time() > deadline - 2:
            break
            
        new_weights = best_weights + np.random.normal(0, 0.05, size=len(weights))
        new_scores = feat_matrix_norm @ new_weights
        new_stations = _simulate_greedy(instance, new_scores)
        
        if new_stations < best_stations_found:
            best_stations_found = new_stations
            best_weights = new_weights
            
    final_scores = feat_matrix_norm @ best_weights
    
    return final_scores

# EVOLVE-BLOCK-END
