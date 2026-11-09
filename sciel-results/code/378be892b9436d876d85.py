"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict, deque

def _simulate_greedy(instance, priorities):
    """
    Simulates the greedy station filling process.
    Returns the number of stations used.
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n_tasks = len(times)
    
    # Build successor map for quick dependency checking
    successors = [[] for _ in range(n_tasks)]
    indegree = [0] * n_tasks
    for i in range(n_tasks):
        for p in preds[i]:
            successors[p].append(i)
            indegree[i] += 1
            
    # Initial eligible tasks: those with no predecessors
    # We use a list for eligible tasks, scanning for max priority that fits.
    # n=30 is small, so O(n) scan per step is fine.
    
    current_station_time = 0.0
    num_stations = 1
    assigned = [False] * n_tasks
    remaining_indegree = indegree.copy()
    
    # Optimization: Pre-compute nothing else, just loop.
    # Note: The greedy rule is "highest-priority eligible task that fits".
    # If none fits, start new station.
    
    while any(not assigned[i] for i in range(n_tasks)):
        # Find eligible tasks
        eligible_tasks = [i for i in range(n_tasks) if not assigned[i] and remaining_indegree[i] == 0]
        
        if not eligible_tasks:
            # Deadlock or finished (should be finished if DAG valid and not all assigned)
            # If not all assigned and no eligible, it's a cycle or error.
            # But task says valid instances.
            # Actually, if we just started a new station, eligible might be empty?
            # No, eligible is based on indegree.
            # If eligible is empty and not all assigned, we can't proceed.
            # However, the standard algorithm handles "no task fits" by moving to next station.
            # But if eligible is empty, there are NO tasks to assign.
            # This implies we are done or stuck.
            # Let's break to avoid infinite loop.
            break
            
        # Sort by priority descending
        # Using sort is O(k log k). With k <= 30, it's fine.
        eligible_tasks.sort(key=lambda x: priorities[x], reverse=True)
        
        # Find the first one that fits
        found = False
        for task in eligible_tasks:
            if current_station_time + times[task] <= cycle_time:
                # Assign task
                assigned[task] = True
                current_station_time += times[task]
                
                # Update successors
                for succ in successors[task]:
                    remaining_indegree[succ] -= 1
                found = True
                break
        
        if not found:
            # Start new station
            num_stations += 1
            current_station_time = 0.0
            # The loop continues. The same eligible tasks are available.
            # In the next iteration, since current_station_time is 0,
            # the highest priority task (which has time <= cycle_time) will fit.
            
    return num_stations

def priority(instance):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n_tasks = len(times)
    
    # Pre-calculate some heuristic values to seed the search
    # 1. Longest Task First (LTF)
    h_ltf = times.copy()
    
    # 2. Longest Processing Time + Successors count (Weighted)
    # Count successors
    succ_count = np.zeros(n_tasks, dtype=int)
    for i in range(n_tasks):
        for p in preds[i]:
            succ_count[p] += 1
    # Simple weighted: time + 0.1 * succ_count
    h_weighted = times + 0.1 * succ_count
    
    # 3. Rank Position Weight (RPW) - approximation
    # Sort by time descending, assign ranks
    sorted_indices = np.argsort(-times)
    ranks = np.zeros(n_tasks)
    for rank, idx in enumerate(sorted_indices):
        ranks[idx] = n_tasks - rank
    h_rpw = times * ranks
    
    # 4. Random
    h_random = np.random.rand(n_tasks)
    
    # Initialize best
    candidates = [h_ltf, h_weighted, h_rpw, h_random]
    best_score = float('inf')
    best_priorities = None
    
    for cand in candidates:
        score = _simulate_greedy(instance, cand)
        if score < best_score:
            best_score = score
            best_priorities = cand.copy()
            
    # If we already found a very good score (e.g., theoretical lower bound), we might stop early?
    # Lower bound is ceil(sum(times) / cycle_time).
    lb = np.ceil(np.sum(times) / cycle_time)
    
    # Local Search: Simulated Annealing / Hill Climbing
    # We will perform a series of restarts.
    # Each restart starts from a random point or a perturbed best.
    # We use "Swaps" as the move operator.
    
    current_priorities = best_priorities.copy()
    
    # Number of restarts and steps depends on time budget.
    # Estimate cost of one simulation: ~30 * 30 ops = 900 ops. Very fast.
    # Python overhead is the main cost. ~1-5 ms per sim.
    # 110s / 5ms = 22,000 sims possible.
    # Let's do 50 restarts, 200 steps each = 10,000 sims. Should be well within time.
    
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break
            
        # Random initialization for this restart
        # Mix of random and best to maintain diversity but converge
        if restart == 0:
            p = best_priorities.copy()
        else:
            # 50% chance to start random, 50% chance to start from best with noise
            if random.random() < 0.5:
                p = np.random.rand(n_tasks)
            else:
                p = best_priorities + np.random.normal(0, 0.1, n_tasks)
                
        score = _simulate_greedy(instance, p)
        if score < best_score:
            best_score = score
            best_priorities = p.copy()
            
        # Local search: 200 steps of swap hill climbing
        for i in range(200):
            if time.time() > deadline - 5.0:
                break
                
            # Pick two random tasks to swap
            idx1, idx2 = random.sample(range(n_tasks), 2)
            
            # Swap priorities
            p[idx1], p[idx2] = p[idx2], p[idx1]
            
            new_score = _simulate_greedy(instance, p)
            
            if new_score <= best_score:
                best_score = new_score
                best_priorities = p.copy()
                # Keep the swap (p is already swapped)
            else:
                # Revert swap
                p[idx1], p[idx2] = p[idx2], p[idx1]
                
            # Optional: If we hit the lower bound, we can stop early for this instance
            if best_score == lb:
                return best_priorities

    return best_priorities
# EVOLVE-BLOCK-END
