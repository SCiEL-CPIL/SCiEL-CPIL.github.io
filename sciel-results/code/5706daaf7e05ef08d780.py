"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from itertools import combinations

def _score_set(U):
    """Compute the score for a finite set U containing 0 using NumPy."""
    U_set = set(U)
    if 0 not in U_set:
        return 0.0
    U_arr = np.array(sorted(U_set))
    n = len(U_arr)
    if n == 0:
        return 0.0
    max_u = int(np.max(U_arr))
    if max_u == 0:
        return 1.0
    
    # Compute sumset and difference set
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
        
    return 1.0 + float(log_ratio) / float(log_denom)

def _exhaustive_search(deadline):
    """Exhaustive search for small sets."""
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # n=3, 4, 5, 6
    # We limit max_val to keep combinatorics manageable
    for n in range(3, 7):
        # Adjust max_val based on n to control C(max_val, n-1)
        max_val = 50 if n <= 4 else (40 if n == 5 else 35)
        
        for combo in combinations(range(1, max_val + 1), n - 1):
            if time.time() > deadline:
                return best_U, best_score
            U = [0] + list(combo)
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
                
    return best_U, best_score

def _local_search(U, deadline, max_iters=1000):
    """Steepest Descent Local Search."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        best_neighbor_score = best_score
        best_neighbor_U = best_U[:]
        
        # 1. Try moving elements
        for i in range(len(U)):
            if U[i] == 0: continue
            old_val = U[i]
            # Search range: 1 to max_u + 10
            for new_val in range(1, max_u + 11):
                if new_val == old_val: continue
                new_U = U[:]
                new_U[i] = new_val
                if len(set(new_U)) != len(new_U): continue
                s = _score_set(new_U)
                if s > best_neighbor_score + 1e-10:
                    best_neighbor_score = s
                    best_neighbor_U = new_U[:]
                    
        # 2. Try adding elements
        for new_val in range(1, max_u + 11):
            if new_val in U: continue
            new_U = U[:] + [new_val]
            new_U.sort()
            s = _score_set(new_U)
            if s > best_neighbor_score + 1e-10:
                best_neighbor_score = s
                best_neighbor_U = new_U[:]
                
        # 3. Try removing elements (if size > 3)
        if len(U) > 3:
            for i in range(1, len(U)):
                new_U = U[:i] + U[i+1:]
                s = _score_set(new_U)
                if s > best_neighbor_score + 1e-10:
                    best_neighbor_score = s
                    best_neighbor_U = new_U[:]
                    
        if best_neighbor_score > best_score + 1e-10:
            best_score = best_neighbor_score
            best_U = best_neighbor_U
            max_u = max(best_U)
        else:
            # Stuck. Try a random perturbation to escape local optimum
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
                        s = _score_set(perturbed)
                        if s > best_score:
                            best_score = s
                            best_U = perturbed
                            max_u = max(best_U)
            else:
                break # No perturbation possible or no improvement, stop
                
    return best_U, best_score

def _get_heuristic_seeds():
    seeds = []
    # Golomb rulers
    seeds.append([0, 1, 3])
    seeds.append([0, 1, 4, 6])
    seeds.append([0, 1, 4, 10, 12])
    seeds.append([0, 1, 4, 10, 12, 17])
    seeds.append([0, 1, 4, 10, 18, 23, 29])
    return seeds

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, keep 5s buffer
    deadline = start_time + 145
    
    # 1. Exhaustive search
    best_U, best_score = _exhaustive_search(deadline)
    
    # 2. Local search from best exhaustive result
    if time.time() < deadline:
        best_U, best_score = _local_search(best_U, deadline)
        
    # 3. Local search from heuristic seeds
    seeds = _get_heuristic_seeds()
    for seed in seeds:
        if time.time() > deadline - 5:
            break
        U_opt, s_opt = _local_search(seed, deadline, max_iters=500)
        if s_opt > best_score:
            best_score = s_opt
            best_U = U_opt

    # 4. Multiple restarts with random initializations
    for _ in range(5): # Up to 5 restarts
        if time.time() > deadline - 2:
            break
        # Generate a random set
        n = random.randint(3, 8)
        max_val = random.randint(10, 40)
        U_rand = [0] + random.sample(range(1, max_val + 1), n - 1)
        U_rand.sort()
        
        # Local search from random start
        U_opt, s_opt = _local_search(U_rand, deadline, max_iters=200)
        if s_opt > best_score:
            best_score = s_opt
            best_U = U_opt
            
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
