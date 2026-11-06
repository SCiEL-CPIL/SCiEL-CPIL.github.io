"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict, deque

def _greedy_balance(times, predecessors, cycle_time, priority_scores):
    """
    Greedy station filling: repeatedly pick the highest-priority eligible task that fits.
    Returns number of stations.
    """
    n = len(times)
    if n == 0:
        return 0
    
    # Build successor map
    successors = defaultdict(list)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Track unassigned predecessor count
    unassigned_preds = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    
    eligible = set(i for i in range(n) if unassigned_preds[i] == 0)
    
    if not eligible and n > 0:
        return 10**9
    
    station_count = 1
    current_load = 0
    
    while eligible:
        # Sort eligible by priority descending
        # For n=30, sorting a list of up to 30 items is fast
        candidates = sorted(eligible, key=lambda i: -priority_scores[i])
        placed_in_this_pass = False
        
        for task in candidates:
            if times[task] <= cycle_time - current_load:
                # Place task
                current_load += times[task]
                eligible.discard(task)
                placed_in_this_pass = True
                
                for succ in successors[task]:
                    unassigned_preds[succ] -= 1
                    if unassigned_preds[succ] == 0:
                        eligible.add(succ)
                break
        
        if not placed_in_this_pass:
            # Overflow to next station
            station_count += 1
            current_load = 0
            # Check for impossible case
            if any(times[t] > cycle_time for t in eligible):
                return 10**9
    
    return station_count

def _compute_critical_path(times, predecessors, successors):
    """Compute longest path from each task to any sink."""
    n = len(times)
    in_degree = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
    
    topo_order = []
    in_deg_copy = in_degree.copy()
    q = deque()
    for i in range(n):
        if in_deg_copy[i] == 0:
            q.append(i)
    
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)
    
    cp = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        cp[u] = times[u]
        for v in successors[u]:
            cp[u] = max(cp[u], times[u] + cp[v])
    
    return cp

def _count_transitive(start, adj_map):
    """Count transitive successors/predecessors."""
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        for s in adj_map[node]:
            if s not in visited:
                visited.add(s)
                stack.append(s)
    return len(visited)

def _simulated_annealing(times, predecessors, cycle_time, initial_priority, deadline):
    """
    Simulated Annealing to optimize the priority vector.
    """
    n = len(times)
    current_priority = initial_priority.copy()
    current_score = _greedy_balance(times, predecessors, cycle_time, current_priority)
    
    best_priority = current_priority.copy()
    best_score = current_score
    
    rng = np.random.RandomState(42)
    
    # Initial temperature
    T = 1.0
    cooling_rate = 0.9995
    min_T = 0.01
    
    start_time = time.time()
    
    while time.time() < deadline:
        if T < min_T:
            # If we cooled down too much, restart with a higher T to escape local minima
            T = 0.5
            # Perturb current solution significantly
            for _ in range(5):
                i, j = rng.choice(n, 2, replace=False)
                current_priority[i], current_priority[j] = current_priority[j], current_priority[i]
            current_score = _greedy_balance(times, predecessors, cycle_time, current_priority)
        
        # Generate neighbor: swap two random priority values
        i, j = rng.choice(n, 2, replace=False)
        new_priority = current_priority.copy()
        new_priority[i], new_priority[j] = current_priority[j], current_priority[i]
        
        new_score = _greedy_balance(times, predecessors, cycle_time, new_priority)
        
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_priority = new_priority
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_priority = current_priority.copy()
        
        T *= cooling_rate
        
        # Optional: If we found a very good solution, we might want to polish
        if best_score == 1:
            break
            
    return best_priority

def _hill_climb_polish(times, predecessors, cycle_time, initial_priority, deadline):
    """
    Best-improvement hill climbing: try all pairs, take the best improvement.
    """
    n = len(times)
    current_priority = initial_priority.copy()
    current_score = _greedy_balance(times, predecessors, cycle_time, current_priority)
    
    best_priority = current_priority.copy()
    best_score = current_score
    
    while time.time() < deadline:
        improved = False
        best_pair = None
        best_pair_score = current_score
        
        # Try all pairs of swaps
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                new_priority = current_priority.copy()
                new_priority[i], new_priority[j] = current_priority[j], new_priority[i]
                new_score = _greedy_balance(times, predecessors, cycle_time, new_priority)
                
                if new_score < best_pair_score:
                    best_pair_score = new_score
                    best_pair = (i, j, new_priority)
        
        if best_pair is not None and best_pair_score < current_score:
            current_priority = best_pair[2]
            current_score = best_pair_score
            improved = True
            
            if current_score < best_score:
                best_score = current_score
                best_priority = current_priority.copy()
        
        if not improved:
            break
            
    return best_priority

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    if n == 0:
        return np.array([])
    
    # Deadline guard
    start_time = time.time()
    # Leave 5s for safety
    deadline = start_time + 115.0  
    
    # Build successor map
    successors = defaultdict(list)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Precompute useful quantities
    # Critical path
    cp = _compute_critical_path(times, predecessors, successors)
    
    # Number of direct successors
    num_direct_succ = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Number of direct predecessors
    num_direct_pred = np.array([len(predecessors[i]) for i in range(n)], dtype=float)
    
    # Transitive successor counts
    trans_succ = np.array([_count_transitive(i, successors) for i in range(n)], dtype=float)
    
    # Transitive predecessor counts
    pred_map = {i: predecessors[i] for i in range(n)}
    trans_pred = np.array([_count_transitive(i, pred_map) for i in range(n)], dtype=float)
    
    # Candidate priority vectors
    candidates = []
    
    # 1. LPT (longest processing time)
    candidates.append(times.copy())
    # 2. Critical path
    candidates.append(cp.copy())
    # 3. Time + critical path
    candidates.append(times + cp)
    # 4. Most direct successors
    candidates.append(num_direct_succ.copy())
    # 5. Fewest direct predecessors
    candidates.append(-num_direct_pred.copy())
    # 6. Time + successors
    candidates.append(times + num_direct_succ)
    # 7. Critical path + successors
    candidates.append(cp + num_direct_succ)
    # 8. Time - predecessors
    candidates.append(times - num_direct_pred)
    # 9. (time * successors) / (preds + 1)
    candidates.append(times * num_direct_succ / (num_direct_pred + 1))
    # 10. cp * num_direct_succ
    candidates.append(cp * num_direct_succ)
    # 11. Transitive successors
    candidates.append(trans_succ.copy())
    # 12. Time + transitive successors
    candidates.append(times + trans_succ)
    # 13. Critical path + transitive successors
    candidates.append(cp + trans_succ)
    # 14. Fewest transitive predecessors
    candidates.append(-trans_pred.copy())
    # 15. Time * transitive successors
    candidates.append(times * trans_succ)
    # 16. Combined: time + 0.5*cp + 0.5*trans_succ
    candidates.append(times + 0.5 * cp + 0.5 * trans_succ)
    # 17. cp * (1 + trans_succ)
    candidates.append(cp * (1 + trans_succ))
    # 18. Weighted: cp + 2*trans_succ + 0.5*times
    candidates.append(cp + 2*trans_succ + 0.5*times)
    # 19. Weighted: times + 2*cp + 0.5*trans_succ
    candidates.append(times + 2*cp + 0.5*trans_succ)

    # Evaluate each candidate and find the best
    best_score = float('inf')
    best_priority = candidates[0]
    
    for pri in candidates:
        if time.time() > deadline:
            break
        n_stations = _greedy_balance(times, predecessors, cycle_time, pri)
        if n_stations < best_score:
            best_score = n_stations
            best_priority = pri.copy()
        if best_score == 1:
            break
    
    # Phase 1: Simulated Annealing for broad exploration
    if time.time() < deadline - 10: # Leave 10s for polish
        best_priority = _simulated_annealing(
            times, predecessors, cycle_time, 
            best_priority, 
            deadline - 10
        )
    
    # Phase 2: Hill Climbing Polish for fine-tuning
    if time.time() < deadline:
        best_priority = _hill_climb_polish(
            times, predecessors, cycle_time, 
            best_priority, 
            deadline
        )
    
    return best_priority
# EVOLVE-BLOCK-END
