import time
import random
import numpy as np
from math import exp, gcd

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 85  # Safety margin for 90s limit

# Pre-compute indices for A+A and A*A (i <= j)
# 20*21/2 = 210 pairs
INDICES = np.triu_indices(MAX_N)

def _objective(A):
    """
    Compute max(|A+A|, |A*A|) for a set A of 20 distinct positive integers.
    A is a list or array of 20 ints.
    """
    A_arr = np.array(A, dtype=np.int64)
    sums = A_arr[INDICES[0]] + A_arr[INDICES[1]]
    products = A_arr[INDICES[0]] * A_arr[INDICES[1]]
    
    # Count unique values
    unique_sums = len(np.unique(sums))
    unique_products = len(np.unique(products))
    
    return max(unique_sums, unique_products)

def _is_valid(A):
    """Check if A has 20 distinct positive integers <= 10000."""
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > MAX_VAL:
            return False
    return True

def _generate_gp(start, p, q, n):
    """
    Generate n distinct integers from a geometric progression:
    a_k = start * (p/q)^k
    Ensures distinctness and bounds.
    """
    A = []
    current = float(start)
    ratio = float(p) / float(q)
    
    for _ in range(n):
        val = int(round(current))
        if val < 1:
            val = 1
        if val > MAX_VAL:
            break
        if val not in A:
            A.append(val)
        current *= ratio
        
    # If we didn't get enough, fill with small increments
    if len(A) < n:
        last = A[-1] if A else start
        while len(A) < n:
            last += 1
            if last > MAX_VAL: 
                break
            if last not in A:
                A.append(last)
            
    return A[:n]

def _local_search_fast(A, time_limit, rng):
    """
    Fast local search using pre-computed indices and incremental updates.
    Uses a mix of small perturbations and structural moves.
    """
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 50.0
    cooling_rate = 0.997
    
    # Pre-compute pair contributions for faster delta calculation
    # For each element i, track which sums/products involve it
    A_arr = np.array(A, dtype=np.int64)
    
    while temperature > 0.01:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random element to modify
        idx = rng.randint(0, 19)
        old_val = A[idx]
        
        # Try a few random changes
        for _ in range(20):
            if time.time() - start_time > time_limit:
                break
            
            # Generate candidate value
            if rng.random() < 0.7:
                # Small perturbation
                scale = max(1, old_val // 20)
                new_val = old_val + rng.randint(-scale, scale)
            else:
                # Larger structural move
                if rng.random() < 0.5:
                    # Move towards creating more collisions
                    # Try values that are multiples/factors of existing elements
                    candidates = []
                    for j in range(20):
                        if j != idx:
                            # Try multiples
                            for mult in [2, 3]:
                                cand = A[j] * mult
                                if 1 <= cand <= MAX_VAL and cand != old_val:
                                    candidates.append(cand)
                            # Try factors
                            for div in [2, 3]:
                                if A[j] % div == 0:
                                    cand = A[j] // div
                                    if 1 <= cand <= MAX_VAL and cand != old_val:
                                        candidates.append(cand)
                    if candidates:
                        new_val = rng.choice(candidates)
                    else:
                        new_val = old_val + rng.randint(-max(1, old_val//5), max(1, old_val//5))
                else:
                    # Random value in a different range
                    new_val = rng.randint(1, MAX_VAL)
            
            if new_val < 1 or new_val > MAX_VAL:
                continue
            if new_val in A:
                continue
            
            # Test the change
            A[idx] = new_val
            new_obj = _objective(A)
            A[idx] = old_val
            
            delta_obj = new_obj - current_obj
            # Accept if better or with probability exp(-delta/T)
            if delta_obj <= 0 or rng.random() < exp(-delta_obj / (temperature + 1e-10)):
                A[idx] = new_val
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
                break
        
        temperature *= cooling_rate
    
    return best_A, best_obj

def _grid_search_gp(start_time, time_budget, rng):
    """
    Deterministic grid search over rational geometric progressions.
    Tries various (start, p, q) combinations.
    """
    best_A = None
    best_obj = float('inf')
    
    # Try various rational ratios p/q
    ratios = []
    for q in range(1, 10):
        for p in range(q + 1, min(q * 2, 20)):
            if gcd(p, q) == 1:
                ratios.append((p, q))
    
    # Also try some special ratios
    ratios.extend([(2, 1), (3, 2), (4, 3), (5, 4), (7, 5), (9, 8)])
    
    # Try various starting values
    starts = [1, 2, 3, 5, 10, 20, 50, 100]
    
    for start in starts:
        for p, q in ratios:
            if time.time() - start_time > time_budget:
                return best_A, best_obj
            
            A = _generate_gp(start, p, q, 20)
            if not _is_valid(A):
                continue
            
            obj = _objective(A)
            if obj < best_obj:
                best_obj = obj
                best_A = list(A)
    
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5  # Leave margin for safety
    
    rng = random.Random(42)
    best_A = None
    best_obj = float('inf')
    
    # Phase 1: Grid search over GP structures (deterministic, fast)
    gp_time = min(30, time_budget * 0.4)
    gp_A, gp_obj = _grid_search_gp(start_time, gp_time, rng)
    
    if gp_A and gp_obj < best_obj:
        best_obj = gp_obj
        best_A = gp_A
    
    # Phase 2: Local search refinement from best GP
    remaining = time_budget - (time.time() - start_time)
    if remaining > 2 and best_A:
        # Run local search from the best GP
        local_time = min(remaining * 0.6, 40)
        A_opt, obj_opt = _local_search_fast(best_A, local_time, rng)
        
        if obj_opt < best_obj:
            best_obj = obj_opt
            best_A = A_opt
    
    # Phase 3: Try a few other structural starts
    remaining = time_budget - (time.time() - start_time)
    other_starts = []
    
    # Arithmetic progression
    other_starts.append(list(range(1, 21)))
    
    # Powers of 2
    other_starts.append([1 << i for i in range(14)] + [8193, 8194, 8195, 8196, 8197, 8198])
    
    # Mixed small AP + large GP
    small_ap = list(range(1, 11))
    large_gp = _generate_gp(100, 2, 1, 10)
    combined = small_ap + [x for x in large_gp if x not in small_ap]
    if len(combined) >= 20:
        other_starts.append(sorted(combined[:20]))
    
    for A in other_starts:
        if not _is_valid(A):
            continue
        
        obj = _objective(A)
        if obj < best_obj:
            best_obj = obj
            best_A = list(A)
    
    # Phase 4: Final refinement
    remaining = time_budget - (time.time() - start_time)
    if remaining > 2 and best_A:
        A_ref, obj_ref = _local_search_fast(best_A, remaining, rng)
        if obj_ref < best_obj:
            best_A = A_ref
            best_obj = obj_ref
    
    if not best_A:
        best_A = list(range(1, 21))
        
    return sorted(best_A)
