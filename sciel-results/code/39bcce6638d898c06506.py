"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
import random

_DEADLINE_SECONDS = 145.0
_START_TIME = time.time()

def _remaining_budget():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)

def _simulate_sgs(instance, order, predecessors, durations, demands, capacities):
    """
    Simulates the serial schedule generation scheme based on a fixed permutation order.
    The 'order' defines the priority: earlier in the list = higher priority.
    Returns the makespan.
    """
    n = len(durations)
    if n == 0:
        return 0.0
    
    # Map activity index to its rank in the priority order
    # Lower rank = higher priority
    rank = np.zeros(n, dtype=int)
    for idx, act in enumerate(order):
        rank[act] = idx
        
    scheduled = [False] * n
    start_times = np.zeros(n, dtype=float)
    
    # Pre-compute successors for quick eligibility check? 
    # Actually, checking predecessors is O(degree). 
    # We need to pick the eligible activity with the lowest rank (highest priority).
    
    scheduled_count = 0
    current_time_lower_bound = 0.0 # Not strictly needed for Serial SGS if we compute earliest start from resources/preds
    
    # To speed up finding the best eligible activity, we can just scan all unscheduled 
    # and check eligibility + rank. N=30 is small.
    
    while scheduled_count < n:
        best_candidate = -1
        best_rank = float('inf')
        
        # Find eligible activity with highest priority (lowest rank)
        for i in range(n):
            if scheduled[i]:
                continue
            # Check predecessors
            eligible = True
            for p in predecessors[i]:
                if not scheduled[p]:
                    eligible = False
                    break
            if eligible:
                if rank[i] < best_rank:
                    best_rank = rank[i]
                    best_candidate = i
                    
        if best_candidate == -1:
            # Should not happen in a valid DAG
            break
            
        j = best_candidate
        
        # Determine earliest start time for j
        # 1. Predecessor constraint
        es_pred = 0.0
        for p in predecessors[j]:
            finish_p = start_times[p] + durations[p]
            if finish_p > es_pred:
                es_pred = finish_p
                
        # 2. Resource constraint
        # Find earliest time T >= es_pred where resources are available
        # Candidate times: es_pred, and end times of all scheduled jobs that use resources demanded by j
        
        candidate_times = {es_pred}
        
        if demands is not None:
            d_j = demands[j]
            for r in range(len(d_j)):
                if d_j[r] > 0:
                    # Add end times of jobs using resource r
                    for k in range(n):
                        if scheduled[k] and demands[k][r] > 0:
                            candidate_times.add(start_times[k] + durations[k])
        
        # Also consider 0.0? es_pred is already >= 0.
        
        earliest_start = float('inf')
        
        # Sort candidate times to find the first feasible one
        for t in sorted(candidate_times):
            if t < es_pred:
                continue
            
            feasible = True
            if demands is not None and capacities is not None:
                for r in range(len(d_j)):
                    if d_j[r] == 0:
                        continue
                    current_load = 0.0
                    for k in range(n):
                        if scheduled[k]:
                            s_k = start_times[k]
                            e_k = s_k + durations[k]
                            if s_k <= t < e_k:
                                current_load += demands[k][r]
                    if current_load + d_j[r] > capacities[r]:
                        feasible = False
                        break
            
            if feasible:
                earliest_start = t
                break
                
        if earliest_start == float('inf'):
            earliest_start = es_pred # Fallback
            
        start_times[j] = earliest_start
        scheduled[j] = True
        scheduled_count += 1
        
    makespan = np.max(start_times + durations) if n > 0 else 0.0
    return makespan

def _generate_initial_order(instance, predecessors, durations, demands, capacities):
    """
    Generates a good initial order using static heuristics.
    """
    n = len(durations)
    
    # Heuristic 1: Longest Processing Time (LPT)
    order_lpt = sorted(range(n), key=lambda x: -durations[x])
    
    # Heuristic 2: Critical Path (Longest path from start to finish in precedence graph)
    # Compute longest path ending at each node (backward) and starting at each node (forward)
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    # Topological sort
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # Forward pass: longest path from start to i
    dist_start = np.zeros(n)
    for u in topo_order:
        for v in successors[u]:
            if dist_start[u] + durations[u] > dist_start[v]:
                dist_start[v] = dist_start[u] + durations[u]
                
    # Backward pass: longest path from i to end
    dist_end = np.zeros(n)
    for u in reversed(topo_order):
        for v in successors[u]:
            if dist_end[v] + durations[v] > dist_end[u]:
                dist_end[u] = dist_end[v] + durations[v]
                
    # Criticality score: dist_start[i] + durations[i] + dist_end[i]
    crit_score = dist_start + durations + dist_end
    order_crit = sorted(range(n), key=lambda x: -crit_score[x])
    
    # Heuristic 3: Resource Weighted Slack
    # Estimate slack based on criticality and resource pressure
    pressure = np.zeros(n)
    if demands is not None and capacities is not None:
        try:
            d_mat = np.asarray(demands, dtype=float)
            c_vec = np.asarray(capacities, dtype=float)
            normalized = d_mat / c_vec
            pressure = np.sum(normalized, axis=1)
        except Exception:
            pressure = np.zeros(n)
            
    # Normalize
    max_crit = np.max(crit_score) if np.max(crit_score) > 0 else 1.0
    max_pres = np.max(pressure) if np.max(pressure) > 0 else 1.0
    
    # Priority: High Criticality, High Pressure
    score = (crit_score / max_crit) + 0.5 * (pressure / max_pres)
    order_rw = sorted(range(n), key=lambda x: -score[x])
    
    # Evaluate these initial orders
    best_order = order_lpt
    best_ms = float('inf')
    
    for o in [order_lpt, order_crit, order_rw]:
        ms = _simulate_sgs(instance, o, predecessors, durations, demands, capacities)
        if ms < best_ms:
            best_ms = ms
            best_order = o
            
    return best_order, best_ms

def priority(instance):
    """
    Returns a priority vector. 
    Since the external SGS picks the highest priority eligible activity, 
    we return a vector where the value corresponds to the rank in the 
    optimized permutation order. 
    Specifically, if activity i is at position k in the optimal order, 
    priority[i] = n - k. This ensures the external SGS picks them in the 
    exact order we simulated.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    n = len(durations)
    if n == 0:
        return np.array([])
        
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
        
    # 1. Generate initial good order
    best_order, best_ms = _generate_initial_order(instance, predecessors, durations, demands, capacities)
    
    # 2. Local Search: Hill Climbing on Permutations
    # We try swapping adjacent elements in the order to see if it improves makespan.
    # This explores the neighborhood of the initial heuristic solution.
    
    current_order = best_order[:]
    current_ms = best_ms
    
    # Time budget management
    # Each simulation is O(N^2 * R) approx. N=30. 
    # We can do several thousand iterations comfortably within 120s.
    
    # Use a random seed for reproducibility if needed, but random swaps are fine.
    # We will perform a limited number of random restarts and hill climbing steps.
    
    max_iterations = 5000 # Upper bound, will break on time
    no_improve_count = 0
    threshold = 100 # If no improvement for 100 steps, try a random restart or larger move
    
    # Pre-allocate lists for speed if possible, but Python lists are okay for N=30
    
    while _remaining_budget() > 10.0:
        if no_improve_count > threshold:
            # Random restart or larger perturbation
            # Swap two random elements
            i, j = random.sample(range(n), 2)
            current_order[i], current_order[j] = current_order[j], current_order[i]
            no_improve_count = 0
        else:
            # Swap adjacent elements
            if n < 2:
                break
            i = random.randint(0, n - 2)
            current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
            
        new_ms = _simulate_sgs(instance, current_order, predecessors, durations, demands, capacities)
        
        if new_ms < current_ms:
            current_ms = new_ms
            no_improve_count = 0
        else:
            # Revert if worse
            if no_improve_count < threshold:
                # Undo the swap
                if random.random() < 0.5: # Heuristic: sometimes keep worse to escape local optima? 
                    # No, standard hill climbing reverts. Let's stick to strict hill climbing first.
                    # Actually, simple hill climbing can get stuck. 
                    # Let's use a "best-so-far" tracking.
                    pass
                # Revert
                if no_improve_count < threshold:
                    # We need to know which swap we did to revert.
                    # Since we modified current_order in place, we should have kept a copy or just revert the specific swap.
                    # In the adjacent swap case:
                    if no_improve_count < threshold:
                        current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
                    else:
                        # Random swap case
                        i, j = random.sample(range(n), 2) # This is wrong, we don't know i,j from previous iteration easily without storing.
                        # Let's restructure to store the move.
                        pass
            no_improve_count += 1
            
            # If we used the random restart logic above, the revert logic is messy.
            # Let's simplify: 
            # 1. Make a copy of current_order.
            # 2. Apply move to copy.
            # 3. Simulate copy.
            # 4. If better, replace current_order with copy.
            # 5. Else, discard copy.
            
            # Since I already modified current_order in the loop above, I must fix the logic.
            # Let's restart the loop logic in the final code.
            
            if _remaining_budget() < 15.0:
                break
                
        max_iterations -= 1
        if max_iterations <= 0:
            break
            
    # If we broke out of the messy loop above, ensure we return a valid priority vector.
    # The logic above had a bug in reverting. Let's write a clean version below.
    
    # --- CLEAN LOCAL SEARCH IMPLEMENTATION ---
    
    # Reset to best found in initial heuristics
    current_order, current_ms = _generate_initial_order(instance, predecessors, durations, demands, capacities)
    
    # Hill Climbing
    no_improve = 0
    while _remaining_budget() > 5.0:
        # Generate a neighbor
        if no_improve > 50:
            # Random restart: pick two random indices and swap
            i, j = random.sample(range(n), 2)
            new_order = current_order[:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
        else:
            # Adjacent swap
            i = random.randint(0, n - 2)
            new_order = current_order[:]
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            
        new_ms = _simulate_sgs(instance, new_order, predecessors, durations, demands, capacities)
        
        if new_ms < current_ms:
            current_order = new_order
            current_ms = new_ms
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve > 200:
            break # Stuck, stop
            
    # Convert best_order to priority vector
    # The external SGS picks max(priority). 
    # If we want activity 'a' to be picked before 'b', priority[a] must be > priority[b].
    # In our 'order', index 0 is highest priority.
    # So priority[order[0]] should be highest.
    # Let priority[order[k]] = n - k.
    
    priority_vec = np.zeros(n, dtype=float)
    for k, act in enumerate(current_order):
        priority_vec[act] = n - k
        
    return priority_vec

# EVOLVE-BLOCK-END
