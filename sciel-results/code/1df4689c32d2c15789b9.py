"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from itertools import combinations

def _score_set_fast(U):
    """Compute score using NumPy for speed."""
    U_arr = np.array(sorted(set(U)), dtype=np.int64)
    n = len(U_arr)
    if n == 0:
        return 0.0
    max_u = int(U_arr[-1])
    if max_u == 0:
        return 1.0
    
    # Compute sumset: all pairwise sums
    sums = U_arr[:, None] + U_arr[None, :]
    sumset_size = len(np.unique(sums))
    
    # Compute difference set: all pairwise differences
    diffs = U_arr[:, None] - U_arr[None, :]
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

def _score_set(U):
    """Fallback score computation (uses fast version)."""
    return _score_set_fast(U)

def _search_small_sets(max_n=6, max_val=40, deadline=None):
    """Exhaustive search for small sets U using fast scoring."""
    best_score = 0.0
    best_U = [0, 1, 3]
    
    for n in range(3, max_n + 1):
        for combo in combinations(range(1, max_val + 1), n - 1):
            if deadline and time.time() > deadline:
                return best_U, best_score
                
            U = [0] + list(combo)
            s = _score_set_fast(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
                
    return best_U, best_score

def _construct_heuristic_sets(deadline=None):
    """Construct sets based on known good structures."""
    candidates = []
    
    # Structure 1: Almost AP
    for k in range(3, 12):
        for M in range(k, 80):
            if deadline and time.time() > deadline:
                break
            U = list(range(k - 1)) + [M]
            if len(set(U)) == len(U):
                candidates.append(U)
                
    # Structure 2: Known Golomb rulers
    known_gr = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 22],
        [0, 1, 4, 10, 12, 17, 22, 29],
    ]
    candidates.extend(known_gr)
    
    # Structure 3: Random perturbations of AP
    random.seed(42)
    for trial in range(100):
        if deadline and time.time() > deadline:
            break
        n = random.randint(3, 10)
        base_ap = list(range(n))
        # Perturb 1-2 elements
        num_perturb = random.randint(1, min(2, n - 1))
        indices = random.sample(range(1, n), num_perturb)
        U = base_ap[:]
        for i in indices:
            new_val = random.randint(1, 100)
            U[i] = new_val
        if len(set(U)) == len(U) and 0 in U:
            candidates.append(sorted(set(U)))
            
    return candidates

def _local_search(U, max_iters=300, deadline=None):
    """Improve set U via local search with expanded neighborhood."""
    U = sorted(set(U))
    best_score = _score_set_fast(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 40
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        
        # Try moving one element to a new value
        for i in range(len(U)):
            old_val = U[i]
            if old_val == 0:
                continue
            # Try a range of new values, including some random jumps
            candidates_new = list(range(1, upper_bound + 1))
            # Add some random values beyond upper_bound
            for _ in range(5):
                candidates_new.append(random.randint(1, upper_bound + 50))
            
            for new_val in candidates_new:
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set_fast(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    upper_bound = max(best_U) + 40
                    improved = True
                    break
            if improved:
                break
        
        # Try removing an element (if n > 2)
        if not improved and len(U) > 2:
            for i in range(1, len(U)):  # Never remove 0
                new_U = U[:i] + U[i+1:]
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set_fast(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    upper_bound = max(best_U) + 40
                    improved = True
                    break
        
        # Try adding a new element
        if not improved and len(U) < 15:
            for new_val in range(1, upper_bound + 10):
                if new_val in U:
                    continue
                new_U = U + [new_val]
                s = _score_set_fast(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    upper_bound = max(best_U) + 40
                    improved = True
                    break
        
        if not improved:
            break
            
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_U = [0, 1, 3]
    best_score = _score_set_fast(best_U)
    
    # 1. Exhaustive search for small sets (faster with NumPy)
    try:
        U_sol, s_sol = _search_small_sets(max_n=5, max_val=60, deadline=deadline)
        if s_sol > best_score:
            best_score = s_sol
            best_U = U_sol
    except Exception:
        pass
        
    # 2. Heuristic candidates
    candidates = _construct_heuristic_sets(deadline=deadline)
    for U in candidates:
        if deadline and time.time() > deadline:
            break
        s = _score_set_fast(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
            
    # 3. Local search with random restarts
    if time.time() < deadline:
        best_U, best_score = _local_search(best_U, max_iters=200, deadline=deadline)
        
        # Random restarts: try local search from different starting points
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            random.seed(123)
            for restart in range(5):
                if time.time() > deadline:
                    break
                # Generate a random starting set
                n = random.randint(3, 8)
                start_set = [0] + sorted(random.sample(range(1, 80), n - 1))
                U_restart, s_restart = _local_search(start_set, max_iters=100, deadline=deadline)
                if s_restart > best_score:
                    best_score = s_restart
                    best_U = U_restart
                    
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
