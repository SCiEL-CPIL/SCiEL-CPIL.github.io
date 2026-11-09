"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import deque
import random

_DEADLINE = 145.0  # safety margin before 150s kill

# Feature definitions
# We will compute these features for each task in each instance.
# Features:
# 0: time
# 1: num_successors
# 2: num_predecessors
# 3: down_time (longest path from task to sink, including task)
# 4: up_time (longest path from source to task, including task)
# 5: criticality (1 if on critical path, else 0) - or distance from critical path
# 6: desc_work (total work in downstream subtree)
# 7: rank (topological order index)
# 8: time / cycle_time
# 9: num_successors / (num_successors + 1)
# 10: down_time / cycle_time

def _compute_features(instance):
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = float(instance.get("cycle_time", 40.0))
    preds = instance.get("predecessors", None)

    # Build adjacency
    succs = [[] for _ in range(n)]
    in_deg = [0] * n
    if preds is not None:
        for i in range(n):
            p = preds[i]
            if p is not None:
                in_deg[i] = len(p)
                for j in p:
                    succs[j].append(i)
    
    # Topological sort (Kahn's)
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    rank = np.zeros(n, dtype=float)
    rank_idx = 0
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        rank[u] = rank_idx
        rank_idx += 1
        for v in succs[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)

    if len(topo_order) != n:
        # Cycle detected, fallback to simple features
        feats = np.zeros((n, 11))
        feats[:, 0] = times
        feats[:, 8] = times / cycle_time
        return feats

    # down_time
    down_time = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        down_time[u] = times[u]
        for v in succs[u]:
            if down_time[v] > 0 or len(succs[u]) > 0:
                down_time[u] = max(down_time[u], times[u] + down_time[v])

    # up_time
    up_time = np.zeros(n, dtype=float)
    for u in topo_order:
        up_time[u] = times[u]
        if preds is not None and preds[u] is not None:
            for v in preds[u]:
                up_time[u] = max(up_time[u], up_time[v] + times[u])

    critical_path = float(np.max(up_time)) if n > 0 else 0.0

    # desc_work
    desc_work = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        desc_work[u] = times[u]
        for v in succs[u]:
            desc_work[u] += desc_work[v]

    # Build feature matrix
    feats = np.zeros((n, 11))
    for i in range(n):
        feats[i, 0] = times[i]
        feats[i, 1] = len(succs[i])
        preds_i = preds[i] if preds is not None and preds[i] is not None else []
        feats[i, 2] = len(preds_i)
        feats[i, 3] = down_time[i]
        feats[i, 4] = up_time[i]
        
        # Criticality: 0 if on critical path, 1 if slightly off, etc.
        # Distance from critical path in terms of time
        # A task is on the critical path if up_time[i] + down_time[i] - times[i] == critical_path
        dist = critical_path - (up_time[i] + down_time[i] - times[i])
        feats[i, 5] = dist / cycle_time  # normalized distance
        
        feats[i, 6] = desc_work[i]
        feats[i, 7] = rank[i] / n
        feats[i, 8] = times[i] / cycle_time
        feats[i, 9] = len(succs[i]) / (len(succs[i]) + 1.0)
        feats[i, 10] = down_time[i] / cycle_time

    return feats, succs, preds, times, cycle_time

def _simulate_assignment(feats, weights, succs, preds, times, cycle_time):
    n = len(times)
    # Compute priority scores
    scores = feats @ weights
    
    # Greedy assignment
    placed = [False] * n
    in_deg = [0] * n
    if preds is not None:
        for i in range(n):
            if preds[i] is not None:
                in_deg[i] = len(preds[i])
    
    # Track available tasks (in_deg == 0 and not placed)
    # We need to efficiently pick the highest priority available task that fits.
    # Since n is small (30), we can just iterate through all available tasks each time.
    
    stations = 0
    current_load = 0.0
    placed_count = 0
    
    while placed_count < n:
        # Find eligible tasks
        best_task = -1
        best_score = -np.inf
        best_fit = False
        
        for i in range(n):
            if not placed[i] and in_deg[i] == 0:
                if times[i] <= cycle_time - current_load:
                    if scores[i] > best_score:
                        best_score = scores[i]
                        best_task = i
                        best_fit = True
        
        if best_task == -1:
            # No task fits in current station?
            # If no task fits, we must start a new station with the smallest task? 
            # Or the highest priority task that fits in an empty station?
            # The problem statement says "stations are filled with the highest-priority eligible task that fits the cycle time".
            # If no task fits in the *current* station, we close it and start a new one.
            # In a new station, current_load is 0, so any task with time <= cycle_time fits.
            # Assuming all tasks have time <= cycle_time (standard ALBP assumption).
            
            # Close current station
            stations += 1
            current_load = 0.0
            
            # Now find the best task for the new empty station
            for i in range(n):
                if not placed[i] and in_deg[i] == 0:
                    if times[i] <= cycle_time:
                        if scores[i] > best_score:
                            best_score = scores[i]
                            best_task = i
                            best_fit = True
            
            if best_task == -1:
                # Should not happen if times are valid
                break
        
        # Place best_task
        placed[best_task] = True
        placed_count += 1
        current_load += times[best_task]
        
        # Update in_deg for successors
        for v in succs[best_task]:
            in_deg[v] -= 1
            
        # If current station is full (no more tasks fit), we don't close it immediately 
        # unless we can't place any more. The loop naturally handles this:
        # If no task fits, we close and restart.
        
    # Don't forget the last station if it has tasks
    if current_load > 0 or placed_count > 0:
        stations += 1
        
    return stations

def _evaluate_weights(weights, instances_data):
    total_stations = 0
    for data in instances_data:
        feats, succs, preds, times, cycle_time = data
        stations = _simulate_assignment(feats, weights, succs, preds, times, cycle_time)
        total_stations += stations
    return total_stations / len(instances_data)

def priority(instance):
    start_time = time.time()
    
    # We need to handle the case where we are called for a NEW instance 
    # that might be one of the 5 used for training, or a new one.
    # The prompt says "Scored by mean number of stations on five generated 30-task instances".
    # This implies the `priority` function is called for each of these 5 instances separately?
    # No, the interface is `priority(instance) -> one score per task`.
    # The evaluator likely calls `priority` 5 times, once for each instance, and measures the result.
    # OR, the evaluator provides one instance, and we return scores.
    # "Scored by mean number of stations on five generated 30-task instances"
    # This suggests the evaluation loop is:
    # for inst in instances:
    #    scores = priority(inst)
    #    simulate(inst, scores)
    #    count_stations
    #    mean += count_stations / 5
    
    # If `priority` is called per instance, we don't have access to the other 4 instances to train on them!
    # Unless... the `instance` object contains all 5? No, "instance has times, predecessors, cycle_time".
    # So we must assume `priority` is called for a single instance.
    
    # Wait, if we can only see one instance at a time, we can't train on 5.
    # However, often in these tasks, the "instance" passed might be a dictionary containing all data, 
    # or the function is expected to be stateful? No, stateless is safer.
    
    # Let's re-read carefully: "Write a Python priority(instance) function that scores each assembly task... Scored by mean number of stations on five generated 30-task instances".
    # This implies the evaluator runs the whole process 5 times.
    # If I cannot see the other instances, I cannot optimize weights for them.
    # I must use a robust heuristic that works well on average.
    
    # BUT, looking at the previous code, it used a fixed heuristic.
    # Is it possible to do local optimization on the *current* instance?
    # Yes! We can optimize the weights *for the current instance* to minimize the number of stations *for that instance*.
    # Since we have ~120s, and n=30, we can run a fast local search on the weights specifically for the instance passed in.
    
    # This is a much better strategy: Instance-specific optimization.
    
    if "times" not in instance:
        return np.zeros(1)
        
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    
    # Pre-compute features and graph structures
    try:
        feats, succs, preds, times_arr, cycle_time = _compute_features(instance)
    except Exception:
        return times.copy()
        
    # Initialize weights with a good heuristic
    # Weights corresponding to:
    # 0: time, 1: num_succ, 2: num_pred, 3: down_time, 4: up_time, 5: crit_dist, 6: desc_work, 7: rank, 8: time_ratio, 9: succ_ratio, 10: down_ratio
    initial_weights = np.array([
        1.0,   # time
        1.0,   # num_successors
        -0.5,  # num_predecessors (fewer preds is better? or more? usually fewer preds means earlier)
        2.0,   # down_time (longer downstream path is critical)
        1.0,   # up_time
        -5.0,  # crit_dist (lower distance is better, so negative weight)
        0.5,   # desc_work
        -0.1,  # rank (earlier rank is better, so negative weight)
        0.5,   # time_ratio
        0.5,   # succ_ratio
        0.5    # down_ratio
    ])
    
    best_weights = initial_weights.copy()
    best_score = _evaluate_weights(best_weights, [(feats, succs, preds, times_arr, cycle_time)])
    
    # Local Search / Hill Climbing
    # We will try to improve the weights by random perturbations.
    # Budget: 100 seconds.
    max_time = 100.0
    step_size = 0.5
    
    # Use Numpy random seed for reproducibility? No, let's use random module.
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < max_time:
        # Generate a perturbation
        # Mix of small and large steps
        if rng.random() < 0.1:
            # Large step
            perturbation = rng.normal(0, step_size * 2, size=len(initial_weights))
        else:
            perturbation = rng.normal(0, step_size, size=len(initial_weights))
            
        new_weights = best_weights + perturbation
        
        new_score = _evaluate_weights(new_weights, [(feats, succs, preds, times_arr, cycle_time)])
        
        if new_score < best_score:
            best_score = new_score
            best_weights = new_weights
            # Reduce step size slightly on success
            step_size *= 0.95
        else:
            # Increase step size slightly on failure? Or keep same.
            # step_size *= 1.01
            pass
            
        # If we have found a very good score, we can stop early?
        # Minimum stations is ceil(total_work / cycle_time) and ceil(critical_path / cycle_time)
        min_possible = max(np.ceil(np.sum(times_arr) / cycle_time), np.ceil(np.max(feats[:, 4]) / cycle_time))
        if best_score <= min_possible:
            break

    # Final check on time
    if time.time() - start_time > _DEADLINE:
        # Return best known weights
        pass

    # Compute final scores
    scores = feats @ best_weights
    
    # Ensure positive? The greedy algorithm just picks max, so relative order matters.
    # But if scores are negative, it's fine as long as we pick max.
    
    return scores

# EVOLVE-BLOCK-END
