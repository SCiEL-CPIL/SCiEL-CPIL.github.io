import time
import math
import random
import numpy as np

def _score_of(U):
    """Calculate the score for a set U."""
    if not U or 0 not in U:
        return 0.0
    U = list(set(U))
    U.sort()
    n = len(U)
    if n < 2:
        return 0.0
    
    # Calculate sumset and difference set sizes
    # Using sets for clarity and speed for small n
    sumset = set()
    diffset = set()
    max_u = U[-1]
    
    for i in range(n):
        for j in range(n):
            sumset.add(U[i] + U[j])
            diffset.add(U[i] - U[j])
            
    size_sum = len(sumset)
    size_diff = len(diffset)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / size_sum
    # Denominator is log(2 * max(U) + 1)
    # If max(U) is 0, log(1) = 0. Handle division by zero.
    denom_arg = 2 * max_u + 1
    if denom_arg <= 1:
        return 0.0
        
    denom = math.log(denom_arg)
    if denom == 0:
        return 0.0
        
    return 1 + math.log(ratio) / denom

def _local_search(U, max_steps=1000, time_limit=10.0):
    """Perform local search starting from U."""
    best_U = list(U)
    best_score = _score_of(best_U)
    
    start_time = time.time()
    n = len(best_U)
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random index to modify (not 0)
        i = random.randint(1, n - 1)
        
        # Determine range for new value
        # Lower bound: U[i-1] + 1
        # Upper bound: U[i+1] - 1 if i < n-1 else best_U[i] + 10 (or some heuristic)
        low = best_U[i-1] + 1
        high = best_U[i+1] - 1 if i < n - 1 else best_U[i] + 10
        
        if low > high:
            # Try to expand if at boundary
            if i == n - 1:
                high = best_U[i] + 10
            else:
                continue # Should not happen if valid state
                
        # Randomly pick a value in [low, high]
        # To allow "jumps", sometimes pick a larger delta
        if random.random() < 0.1 and i == n - 1:
             new_val = best_U[i] + random.randint(1, 20)
        else:
             new_val = random.randint(low, high)
        
        # Construct new set
        new_U = best_U.copy()
        new_U[i] = new_val
        new_U.sort()
        
        # Check validity (uniqueness)
        # Since we picked from (U[i-1], U[i+1]), and U was unique, 
        # new_val might collide with other elements if not careful?
        # Actually, if we replace U[i] with a value between neighbors, 
        # it is distinct from neighbors. But what about non-neighbors?
        # In a sorted list, if U[i-1] < new_val < U[i+1], 
        # then new_val is distinct from U[i-1] and U[i+1].
        # Since U[k] <= U[i-1] for k < i-1 and U[k] >= U[i+1] for k > i+1,
        # new_val is distinct from all.
        
        new_score = _score_of(new_U)
        
        if new_score > best_score:
            best_score = new_score
            best_U = new_U
        else:
            # Optional: Simulated annealing style acceptance?
            # For now, strict hill climbing.
            pass
            
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit to be safe
    
    best_global_U = [0, 1, 3]
    best_global_score = _score_of(best_global_U)
    
    # Known optimal Golomb rulers
    golomb_rulers = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 25],
        [0, 1, 4, 10, 12, 17, 25, 34],
        [0, 1, 4, 10, 12, 17, 25, 34, 42],
        [0, 1, 4, 10, 12, 17, 25, 34, 42, 52],
        [0, 1, 4, 10, 12, 17, 25, 34, 42, 52, 60], # Approx for n=11
        [0, 1, 4, 10, 12, 17, 25, 34, 42, 52, 60, 72] # Approx for n=12
    ]
    
    # Evaluate initial candidates
    for U in golomb_rulers:
        if time.time() > deadline:
            break
        s = _score_of(U)
        if s > best_global_score:
            best_global_score = s
            best_global_U = U
    
    # Local search on each Golomb ruler
    for U in golomb_rulers:
        if time.time() > deadline:
            break
        
        # Reduce time limit per search as we go
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        t_limit = min(5.0, remaining / 2)
        
        U_opt, s_opt = _local_search(U, max_steps=500, time_limit=t_limit)
        
        if s_opt > best_global_score:
            best_global_score = s_opt
            best_global_U = U_opt
    
    # Try some random restarts with different n
    for n in range(4, 15):
        if time.time() > deadline:
            break
            
        remaining = deadline - time.time()
        if remaining < 10:
            break
            
        # Random initial set
        # Start with 0, then pick random increasing integers
        U_rand = [0]
        curr = 0
        for _ in range(n - 1):
            curr += random.randint(1, 10)
            U_rand.append(curr)
            
        t_limit = min(5.0, remaining / 2)
        U_opt, s_opt = _local_search(U_rand, max_steps=500, time_limit=t_limit)
        
        if s_opt > best_global_score:
            best_global_score = s_opt
            best_global_U = U_opt

    return best_global_U
