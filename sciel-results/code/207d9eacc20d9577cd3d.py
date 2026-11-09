import numpy as np
import time
import random
from collections import deque

def _greedy_stations(times, preds_list, succs, order, prio, ct):
    n = len(times)
    remaining = [0] * n
    for j in range(n):
        for i in preds_list[j]:
            remaining[j] += 1
    
    unassigned = set(range(n))
    station = 0
    
    while unassigned:
        station += 1
        load = 0.0
        # Iterate through all unassigned tasks to find eligible ones
        # We don't maintain a dynamic 'eligible' list in the inner loop to avoid stale state issues,
        # we just check remaining[i] == 0.
        # To speed up, we can iterate over a sorted list of unassigned by priority, 
        # but rebuilding that list each station is O(n log n).
        # Given n=30, O(n^2) is fine.
        
        # We need to pick the highest priority eligible task that fits.
        # Let's collect eligible tasks first.
        eligible_tasks = []
        for i in unassigned:
            if remaining[i] == 0:
                eligible_tasks.append(i)
                
        if not eligible_tasks:
            # Should not happen if graph is valid, but safety break
            break

        # Sort eligible tasks by priority (descending)
        # We use prio[i] as the key.
        # Note: The problem says "stations are filled with the highest-priority eligible task that fits".
        # This implies a sequential filling process.
        # We must simulate the filling correctly.
        
        # To do this efficiently, we can just iterate through all eligible tasks sorted by priority.
        # If a task fits, we take it.
        # We need to be careful: taking a task might make others eligible? No, predecessors must be done.
        # So the set of eligible tasks only grows as we assign tasks?
        # Actually, assigning a task i decrements remaining for succs[i].
        # So new tasks might become eligible.
        # However, the standard interpretation of "greedy station filling" usually implies:
        # 1. Identify all currently eligible tasks.
        # 2. Sort them by priority.
        # 3. Pick the first one that fits.
        # 4. Assign it.
        # 5. Repeat until no more eligible tasks fit.
        
        # Let's implement this loop.
        assigned_in_station = True
        while assigned_in_station:
            assigned_in_station = False
            # Re-evaluate eligible tasks
            current_eligible = [i for i in unassigned if remaining[i] == 0]
            if not current_eligible:
                break
            
            # Sort by priority descending
            # Tie-breaker: larger time? or index? 
            # The problem doesn't specify tie-breaking for priority, but usually higher priority is better.
            # Let's sort by (-prio[i], -times[i]) to be deterministic and favor larger tasks if prio is equal?
            # Or just -prio[i]. Let's stick to -prio[i] and then -times[i] to pack larger items?
            # Actually, for balancing, putting larger items first is often better (LPT).
            # But priority is the key.
            
            # To allow the local search to work, prio[i] is a float.
            # Sort indices
            current_eligible.sort(key=lambda i: (-prio[i], -times[i]))
            
            for i in current_eligible:
                if times[i] <= ct - load:
                    # Assign task i
                    load += times[i]
                    unassigned.discard(i)
                    for v in succs[i]:
                        remaining[v] -= 1
                    assigned_in_station = True
                    break # Restart the while loop to re-evaluate eligibility
                    
    return station

def _topo(times, preds_list, succs):
    n = len(times)
    indeg = [0] * n
    for j in range(n):
        indeg[j] = len(preds_list[j])
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) < n:
        order.extend([i for i in range(n) if i not in set(order)])
    return order

def priority(instance):
    start = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    times = np.asarray(instance["times"], dtype=float)
    preds_raw = instance["predecessors"]
    preds_list = [list(p) for p in preds_raw]
    n = len(times)
    if n == 0:
        return np.array([])

    ct = float(instance.get("cycle_time", 40.0))

    succs = [[] for _ in range(n)]
    for j in range(n):
        for i in preds_list[j]:
            succs[i].append(j)

    order = _topo(times, preds_list, succs)
    ridx = {i: k for k, i in enumerate(order)}
    rorder = order[::-1]

    # Positional weight
    pw = times.copy()
    for u in rorder:
        s = times[u]
        for v in succs[u]:
            s += pw[v]
        pw[u] = s

    # Level
    level = np.zeros(n, dtype=float)
    for u in order:
        m = 0.0
        for i in preds_list[u]:
            if level[i] > m:
                m = level[i]
        level[u] = m + 1.0

    succ_count = np.array([len(succs[i]) for i in range(n)], dtype=float)

    # Initial Priority: Combination of structural features
    # High PW, High Level, Low Succ Count (sinks first?), High Time?
    # Let's use a weighted sum.
    # Normalizing helps?
    # Let's just use raw values scaled.
    prio = pw + 0.1 * level + 0.01 * (n - succ_count) + 0.001 * times
    
    # Local Search / Hill Climbing
    # Objective: Minimize number of stations
    current_score = _greedy_stations(times, preds_list, succs, order, prio, ct)
    best_prio = prio.copy()
    best_score = current_score
    
    # Random seed for reproducibility within run?
    # Using time-based seed is fine.
    rng = random.Random(42) # Fixed seed for stability? Or time? 
    # Let's use a fixed seed to ensure deterministic behavior for the same input, 
    # which is often preferred in these tasks unless specified otherwise.
    
    max_iterations = 50000 # Plenty of iterations for n=30
    
    while time.time() < deadline and max_iterations > 0:
        max_iterations -= 1
        
        # Perturbation: Swap priorities of two random tasks
        i, j = rng.sample(range(n), 2)
        
        # Swap
        temp = prio[i]
        prio[i] = prio[j]
        prio[j] = temp
        
        new_score = _greedy_stations(times, preds_list, succs, order, prio, ct)
        
        if new_score <= best_score:
            best_score = new_score
            best_prio = prio.copy()
        else:
            # Revert
            prio[i] = prio[j]
            prio[j] = temp
            
        # Optional: Add small noise to escape plateaus?
        # If we are stuck, we might want to do a random walk.
        # But simple hill climbing on swaps is often strong for n=30.
        
    return best_prio
