import math
import time
import random
import numpy as np
from itertools import combinations

def score_of(U):
    if not U:
        return 0.0
    U_sorted = sorted(set(U))
    max_u = max(U_sorted)
    if max_u == 0:
        return 0.0
    sums = set()
    diffs = set()
    for a in U_sorted:
        for b in U_sorted:
            sums.add(a + b)
            diffs.add(a - b)
    num_sums = len(sums)
    num_diffs = len(diffs)
    if num_sums == 0:
        return 0.0
    ratio = num_diffs / num_sums
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0.0
    return 1 + math.log(ratio) / denom

def _score_set_fast(U):
    """Compute score using numpy for speed."""
    U_arr = np.array(sorted(set(U)), dtype=np.int64)
    n = len(U_arr)
    if n == 0:
        return 0.0
    max_u = np.max(U_arr)
    if max_u == 0:
        return 1.0
    
    sums = np.add.outer(U_arr, U_arr)
    diffs = np.subtract.outer(U_arr, U_arr)
    
    unique_sums = np.unique(sums)
    unique_diffs = np.unique(diffs)
    
    len_sums = len(unique_sums)
    len_diffs = len(unique_diffs)
    
    if len_sums == 0:
        return 0.0
        
    ratio = len_diffs / len_sums
    if ratio <= 0:
        return 0.0
        
    log_ratio = np.log(ratio)
    log_denom = np.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_denom

def _get_golomb_rulers():
    """
    Return a list of known optimal Golomb rulers for small n.
    These maximize |U-U| for a given n and minimize max(U).
    """
    rulers = [
        [0],
        [0, 1],
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 18, 23],
        [0, 1, 4, 10, 18, 23, 29],
        [0, 1, 4, 10, 18, 23, 29, 36]
    ]
    return rulers

def _local_search(U, deadline, max_iters=1000, step_size=None):
    """
    Local search with random restarts and simulated annealing-like acceptance.
    """
    U = sorted(set(U))
    best_score = _score_set_fast(U)
    best_U = U[:]
    max_u = max(U)
    
    # Simple steepest descent first
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        best_neighbor_score = best_score
        best_neighbor_U = best_U[:]
        
        # 1. Move elements
        for i in range(len(U)):
            if U[i] == 0: continue
            old_val = U[i]
            # Search range
            search_range = range(1, max_u + 5)
            for new_val in search_range:
                if new_val == old_val: continue
                new_U = U[:]
                new_U[i] = new_val
                if len(set(new_U)) != len(new_U): continue
                s = _score_set_fast(new_U)
                if s > best_neighbor_score + 1e-10:
                    best_neighbor_score = s
                    best_neighbor_U = new_U[:]
                    
        # 2. Add elements (if size < 10)
        if len(U) < 10:
            for new_val in range(1, max_u + 5):
                if new_val in U: continue
                new_U = U[:] + [new_val]
                new_U.sort()
                s = _score_set_fast(new_U)
                if s > best_neighbor_score + 1e-10:
                    best_neighbor_score = s
                    best_neighbor_U = new_U[:]
                    
        # 3. Remove elements (if size > 3)
        if len(U) > 3:
            for i in range(1, len(U)):
                new_U = U[:i] + U[i+1:]
                s = _score_set_fast(new_U)
                if s > best_neighbor_score + 1e-10:
                    best_neighbor_score = s
                    best_neighbor_U = new_U[:]
                    
        if best_neighbor_score > best_score + 1e-10:
            best_score = best_neighbor_score
            best_U = best_neighbor_U
            max_u = max(best_U)
        else:
            # Stuck. Try random perturbation
            if len(best_U) > 3:
                idx = random.randint(1, len(best_U) - 1)
                old_val = best_U[idx]
                new_val = random.randint(1, max_u + 10)
                if new_val not in best_U:
                    perturbed = best_U[:]
                    perturbed[idx] = new_val
                    perturbed.sort()
                    perturbed = list(set(perturbed))
                    if 0 in perturbed:
                        s = _score_set_fast(perturbed)
                        if s > best_score:
                            best_score = s
                            best_U = perturbed
                            max_u = max(best_U)
            else:
                break
                
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_U = [0, 1, 3]
    best_score = _score_set_fast(best_U)
    
    # 1. Try known Golomb rulers
    for ruler in _get_golomb_rulers():
        if time.time() > deadline:
            break
        if len(ruler) < 3: continue
        U_opt, s_opt = _local_search(ruler, deadline, max_iters=200)
        if s_opt > best_score:
            best_score = s_opt
            best_U = U_opt
            
    # 2. Exhaustive search for small n
    # Focus on n=4, 5, 6 with small max values
    for n in range(4, 7):
        if time.time() > deadline:
            break
        max_val = 30
        for combo in combinations(range(1, max_val + 1), n - 1):
            if time.time() > deadline:
                break
            U = [0] + list(combo)
            s = _score_set_fast(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
                
    # 3. Random restarts
    for _ in range(10):
        if time.time() > deadline:
            break
        n = random.randint(4, 8)
        max_val = random.randint(15, 40)
        U_rand = [0] + random.sample(range(1, max_val + 1), n - 1)
        U_rand.sort()
        U_opt, s_opt = _local_search(U_rand, deadline, max_iters=100)
        if s_opt > best_score:
            best_score = s_opt
            best_U = U_opt
            
    return sorted(set(best_U))
