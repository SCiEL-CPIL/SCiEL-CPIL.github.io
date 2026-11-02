import time
import math
import random
import numpy as np
from itertools import combinations

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U_arr = np.array(sorted(set(U)), dtype=np.int64)
    n = len(U_arr)
    if n == 0:
        return 0.0
    max_u = int(U_arr[-1])
    if max_u == 0:
        return 1.0
    
    # Compute sumset and difference set using numpy broadcasting
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    
    sumset_size = len(np.unique(sums))
    diffset_size = len(np.unique(diffs))
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    if ratio == 1.0:
        return 1.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_denom

def _exhaustive_search(max_n=5, max_val=100, deadline=None):
    """Exhaustive search for small sets U."""
    best_score = 0.0
    best_U = [0, 1, 4] # Default to a known good set
    
    for n in range(3, max_n + 1):
        # For larger n, reduce max_val to keep search space manageable
        current_max_val = max_val
        if n == 4:
            current_max_val = 60
        elif n >= 5:
            current_max_val = 30
            
        for combo in combinations(range(1, current_max_val + 1), n - 1):
            if deadline and time.time() > deadline:
                return best_U, best_score
                
            U = [0] + list(combo)
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
                
    return best_U, best_score

def _local_search(U, max_iters=500, deadline=None):
    """Improve set U via local search."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 20
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        
        # Try moving one element
        for i in range(len(U)):
            old_val = U[i]
            if old_val == 0:
                continue
            for new_val in range(1, upper_bound + 1):
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
            if improved:
                break
        
        # If no move improved, try adding an element
        if not improved and len(best_U) < 6:
            for new_val in range(1, upper_bound + 1):
                if new_val in best_U:
                    continue
                new_U = best_U + [new_val]
                new_U.sort()
                s = _score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U
                    improved = True
                    break
        
        # If still no improvement, try removing an element
        if not improved and len(best_U) > 3:
            for i in range(1, len(best_U)):  # Don't remove 0
                new_U = best_U[:i] + best_U[i+1:]
                s = _score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U
                    improved = True
                    break
        
        if not improved:
            break
            
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    best_U = [0, 1, 4]
    best_score = _score_set(best_U)
    
    # 1. Exhaustive search for small sets
    try:
        U_sol, s_sol = _exhaustive_search(max_n=4, max_val=100, deadline=deadline)
        if s_sol > best_score:
            best_score = s_sol
            best_U = U_sol
    except Exception:
        pass
    
    # 2. Local search on best found
    if time.time() < deadline:
        best_U, best_score = _local_search(best_U, max_iters=100, deadline=deadline)
        
    return sorted(set(best_U))
