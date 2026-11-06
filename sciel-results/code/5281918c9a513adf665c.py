import time
import random
from itertools import combinations

def count_conflicts(S):
    """Count the number of unordered pair-sum collisions in S.
    A collision is two distinct pairs (i,j) and (k,l) with i<=j, k<=l,
    (i,j) != (k,l), and S[i]+S[j] == S[k]+S[l].
    More collisions means the set is 'harder' to extract a large Sidon subset from."""
    sums = {}
    n = len(S)
    for i in range(n):
        si = S[i]
        for j in range(i, n):
            s = si + S[j]
            if s in sums:
                sums[s] += 1
            else:
                sums[s] = 1
    # Total collisions = sum over all sums of C(count, 2)
    total = 0
    for c in sums.values():
        if c > 1:
            total += c * (c - 1) // 2
    return total

def local_search(S, max_iter=5000, time_budget=None, start_time=None):
    """Stochastic local search with Simulated Annealing and perturbations."""
    S = sorted(S)
    current_conflicts = count_conflicts(S)
    best_S = list(S)
    best_conflicts = current_conflicts
    n = len(S)
    universe = list(range(129))
    universe_set = set(universe)
    
    # Simulated Annealing parameters
    temp = 10.0
    cooling_rate = 0.995
    min_temp = 0.1
    stagnation_limit = 200
    stagnation_count = 0
    
    for it in range(max_iter):
        if time_budget is not None and start_time is not None:
            if time.time() - start_time > time_budget:
                break
        
        # Generate a move
        # 10% chance of perturbation (replace 2-3 elements)
        if random.random() < 0.1:
            num_replacements = random.randint(2, 3)
            indices_to_replace = random.sample(range(n), num_replacements)
            old_vals = [S[i] for i in indices_to_replace]
            for i in indices_to_replace:
                # Find a new value not in S
                while True:
                    new_val = random.choice(universe)
                    if new_val not in S:
                        break
                S[i] = new_val
            
            new_conflicts = count_conflicts(S)
            
            if new_conflicts > current_conflicts:
                current_conflicts = new_conflicts
                stagnation_count = 0
            else:
                # Accept bad move with probability exp(-delta/T)
                delta = current_conflicts - new_conflicts
                if random.random() < pow(2.71828, -delta / temp):
                    current_conflicts = new_conflicts
                    stagnation_count += 1
                else:
                    # Revert
                    for i, val in zip(indices_to_replace, old_vals):
                        S[i] = val
                    stagnation_count += 1
        else:
            # Standard single swap or replace
            if random.random() < 0.5:
                # Swap two elements
                i, j = random.sample(range(n), 2)
                S[i], S[j] = S[j], S[i]
                new_conflicts = count_conflicts(S)
                
                if new_conflicts > current_conflicts:
                    current_conflicts = new_conflicts
                    stagnation_count = 0
                else:
                    delta = current_conflicts - new_conflicts
                    if random.random() < pow(2.71828, -delta / temp):
                        current_conflicts = new_conflicts
                        stagnation_count += 1
                    else:
                        S[i], S[j] = S[j], S[i]
                        stagnation_count += 1
            else:
                # Replace one element
                i = random.randrange(n)
                old_val = S[i]
                while True:
                    new_val = random.choice(universe)
                    if new_val not in S:
                        break
                S[i] = new_val
                new_conflicts = count_conflicts(S)
                
                if new_conflicts > current_conflicts:
                    current_conflicts = new_conflicts
                    stagnation_count = 0
                else:
                    delta = current_conflicts - new_conflicts
                    if random.random() < pow(2.71828, -delta / temp):
                        current_conflicts = new_conflicts
                        stagnation_count += 1
                    else:
                        S[i] = old_val
                        stagnation_count += 1

        # Update best
        if current_conflicts > best_conflicts:
            best_conflicts = current_conflicts
            best_S = list(S)
        
        # Cooling
        temp = max(min_temp, temp * cooling_rate)
        
        # If stagnated for too long, do a hard reset to best and restart search from there? 
        # Or just increase perturbation rate? 
        # For simplicity, we just let SA handle it, but if temp is min and we are stuck, 
        # we might want to kick.
        if temp == min_temp and stagnation_count > stagnation_limit:
             # Kick: replace 3 random elements with random values
             indices_to_replace = random.sample(range(n), 3)
             old_vals = [S[i] for i in indices_to_replace]
             for i in indices_to_replace:
                 while True:
                     new_val = random.choice(universe)
                     if new_val not in S:
                         break
                 S[i] = new_val
             current_conflicts = count_conflicts(S)
             stagnation_count = 0
             temp = 5.0 # Reheat slightly

    return best_S, best_conflicts

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_S = None
    best_conflicts = -1
    
    # Initial candidates: try several structured starts
    initial_sets = []
    # Consecutive
    initial_sets.append(list(range(12)))
    # Arithmetic progression with large step
    initial_sets.append([i * 11 for i in range(12)])
    # Random starts
    for _ in range(15):
        initial_sets.append(sorted(random.sample(range(129), 12)))
    
    for S0 in initial_sets:
        if time.time() > deadline:
            break
        S, conflicts = local_search(S0, max_iter=3000, time_budget=deadline - time.time(), start_time=start_time)
        if conflicts > best_conflicts:
            best_conflicts = conflicts
            best_S = S
    
    # Additional random restarts if time remains
    while time.time() < deadline - 2.0:
        S0 = sorted(random.sample(range(129), 12))
        S, conflicts = local_search(S0, max_iter=2000, time_budget=deadline - time.time() - 2.0, start_time=start_time)
        if conflicts > best_conflicts:
            best_conflicts = conflicts
            best_S = S
    
    if best_S is None:
        return list(range(12))
    
    return sorted(best_S)
