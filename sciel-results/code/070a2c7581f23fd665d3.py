"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

def _count_stations(order, times, successors, cycle_time):
    """
    Given a topological order of tasks, simulate the greedy packing into stations.
    Returns the number of stations used.
    """
    n = len(order)
    assigned = [False] * n
    # Map original index to position in order? No, we need to know eligibility.
    # Eligibility depends on predecessors being assigned.
    # We iterate through the 'order' list. 
    # Wait, the greedy algorithm picks the highest priority eligible task.
    # If we provide a fixed 'order' (permutation), the greedy algorithm will pick the first task in that permutation that is eligible and fits.
    # So, we simulate:
    # 1. Maintain set of eligible tasks.
    # 2. Iterate through 'order'. 
    # 3. If task is eligible and fits in current station, assign it.
    # 4. If no task in order fits, start new station.
    
    # To make this efficient, we need to quickly check if a task is eligible.
    # We can precompute remaining preds.
    # But 'order' is a permutation of all tasks.
    # Let's simulate step-by-step.
    
    # Pre-compute pred counts
    # We need pred_sets or counts. Let's pass them or compute inside? 
    # For speed, let's assume we have a global structure or pass necessary data.
    # Let's modify signature to take pred_counts and successors.
    pass

def priority(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build graph structures
    pred_sets = [set(p) for p in predecessors]
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in pred_sets[i]:
            successors[p].append(i)
    
    # Topological sort to get a valid initial order
    in_degree = [len(p) for p in pred_sets]
    queue = [i for i in range(n) if in_degree[i] == 0]
    top_order = []
    temp_in_deg = in_degree[:]
    while queue:
        # Sort queue by time descending for a good initial heuristic (LPT)
        queue.sort(key=lambda x: times[x], reverse=True)
        u = queue.pop(0)
        top_order.append(u)
        for v in successors[u]:
            temp_in_deg[v] -= 1
            if temp_in_deg[v] == 0:
                queue.append(v)
    
    if len(top_order) != n:
        # Cycle detected or error, fallback to simple sort
        top_order = sorted(range(n), key=lambda x: times[x], reverse=True)

    def simulate(order):
        """
        Simulates the greedy line balancing given a priority order.
        The greedy rule: Look at eligible tasks. Pick the one with highest priority (earliest in 'order') that fits.
        """
        # Map task index to its rank in the order (lower rank = higher priority)
        rank = [0] * n
        for i, task in enumerate(order):
            rank[task] = i
        
        remaining_preds = [len(p) for p in pred_sets]
        eligible = [i for i in range(n) if remaining_preds[i] == 0]
        
        stations = 1
        current_load = 0.0
        assigned = [False] * n
        assigned_count = 0
        
        while assigned_count < n:
            # Find the best eligible task that fits
            # We need the eligible task with the smallest rank (highest priority) that fits
            best_task = -1
            best_rank = float('inf')
            
            # Optimization: Instead of checking all eligible tasks every time, 
            # we could maintain a heap, but n=30 is small. 
            # However, we iterate through ALL eligible tasks to find the min rank that fits.
            for t in eligible:
                if assigned[t]:
                    continue
                if current_load + times[t] <= cycle_time:
                    if rank[t] < best_rank:
                        best_rank = rank[t]
                        best_task = t
            
            if best_task != -1:
                # Assign best_task
                assigned[best_task] = True
                assigned_count += 1
                current_load += times[best_task]
                
                # Update successors
                for s in successors[best_task]:
                    remaining_preds[s] -= 1
                    if remaining_preds[s] == 0:
                        eligible.append(s)
            else:
                # No task fits. Start new station.
                stations += 1
                current_load = 0.0
                # If we are stuck (no eligible task fits even in empty station), break to avoid infinite loop
                # This shouldn't happen if max(time) <= cycle_time
                if current_load == 0:
                    # Check if any eligible task exists at all
                    if not eligible:
                        break
                    # If eligible exists but none fit in empty station, it's an invalid instance or time > cycle_time
                    # Force assign smallest to break
                    min_t = min(eligible, key=lambda x: times[x])
                    assigned[min_t] = True
                    assigned_count += 1
                    current_load = times[min_t]
                    for s in successors[min_t]:
                        remaining_preds[s] -= 1
                        if remaining_preds[s] == 0:
                            eligible.append(s)

        return stations

    # Initial solution
    current_order = top_order[:]
    current_cost = simulate(current_order)
    best_order = current_order[:]
    best_cost = current_cost
    
    # Lower bound
    lower_bound = int(math.ceil(np.sum(times) / cycle_time))
    
    if best_cost <= lower_bound:
        # Convert best_order to a score vector
        scores = np.zeros(n)
        for i, task in enumerate(best_order):
            scores[task] = n - i # Higher score for higher priority
        return scores

    # Local Search: Random Swap Hill Climbing
    # We want to minimize stations.
    # We will randomly swap two tasks in the order and see if it improves.
    # To ensure validity, we don't need to check topological validity of the order itself 
    # because the simulator handles eligibility dynamically. 
    # The 'order' just defines the priority tie-breaker/pick order.
    
    random.seed(42)
    max_iters = 50000 # We can do many fast iterations
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
        
        # Pick two random indices to swap
        idx1 = random.randint(0, n - 1)
        idx2 = random.randint(0, n - 1)
        
        if idx1 == idx2:
            continue
            
        # Swap
        current_order[idx1], current_order[idx2] = current_order[idx2], current_order[idx1]
        
        new_cost = simulate(current_order)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_order = current_order[:]
                if best_cost <= lower_bound:
                    break
        else:
            # Revert swap
            current_order[idx1], current_order[idx2] = current_order[idx2], current_order[idx1]
            
            # Optional: If stuck for a while, try a random restart or larger perturbation
            # For simplicity, just keep swapping.
            # If we want to escape local minima, we could allow worse moves (Simulated Annealing),
            # but for n=30 and integer objective, simple hill climbing with restarts might be better.
            
            # Let's add a simple restart mechanism if no improvement for 1000 steps
            # Actually, let's just run until deadline.
    
    # Convert best_order to score vector
    scores = np.zeros(n)
    for i, task in enumerate(best_order):
        scores[task] = n - i
        
    return scores
# EVOLVE-BLOCK-END
