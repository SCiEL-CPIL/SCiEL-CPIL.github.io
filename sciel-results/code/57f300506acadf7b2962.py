"""Integer sum-product extremal sets - minimize max(|A+A|, |A*A|)"""
import time
import random
import math
import numpy as np

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    A_arr = np.array(A, dtype=np.int64)
    n = len(A_arr)
    
    # Sumset with equal summands allowed (i <= j)
    i_idx, j_idx = np.triu_indices(n, k=0)
    sums = A_arr[i_idx] + A_arr[j_idx]
    n_sums = len(np.unique(sums))
    
    # Product set with equal factors allowed (i <= j)
    products = A_arr[i_idx] * A_arr[j_idx]
    n_products = len(np.unique(products))
    
    return max(n_sums, n_products)

def _is_valid(A):
    """Check if A has 20 distinct positive integers <= 10000."""
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > 10000:
            return False
    return True

def _generate_gp(ratio, start_val):
    """Generate a geometric progression set, filling gaps if needed."""
    A = set()
    current = start_val
    while len(A) < 20 and current <= 10000:
        A.add(current)
        current = int(current * ratio)
        if current <= start_val:
            break
    
    if len(A) < 20:
        # Fill remaining with random small integers to maintain distinctness
        candidates = list(range(1, 10001))
        random.shuffle(candidates)
        for v in candidates:
            if len(A) >= 20:
                break
            if v not in A:
                A.add(v)
    return list(A)

def _generate_rational_gp(p, q, start_val):
    """Generate a geometric progression with rational ratio p/q."""
    A = set()
    current = start_val
    while len(A) < 20 and current <= 10000:
        A.add(current)
        new_val = (current * p) // q
        if new_val <= current:
            break
        current = new_val
    
    if len(A) < 20:
        # Fill remaining with random values
        candidates = list(range(1, 10001))
        random.shuffle(candidates)
        for v in candidates:
            if len(A) >= 20:
                break
            if v not in A:
                A.add(v)
    return list(A)

def _local_search(A, time_limit, rng):
    """Local search with simulated annealing to minimize the objective."""
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 50.0
    cooling_rate = 0.99
    min_temperature = 0.01
    
    while temperature > min_temperature:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random element to modify
        idx = rng.randint(0, 19)
        old_val = A[idx]
        
        # Try a few random changes
        found_improvement = False
        for _ in range(30):
            if time.time() - start_time > time_limit:
                break
            
            # Random delta - biased toward smaller changes for refinement
            if rng.random() < 0.4:
                new_val = old_val + rng.randint(-20, 20)
            elif rng.random() < 0.7:
                new_val = old_val + rng.randint(-100, 100)
            else:
                new_val = old_val + rng.randint(-500, 500)
            
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in A:
                continue
            
            # Test the change
            A[idx] = new_val
            new_obj = _objective(A)
            A[idx] = old_val
            
            delta = new_obj - current_obj
            if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                A[idx] = new_val
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
                found_improvement = True
                break
        
        temperature *= cooling_rate
    
    return best_A, best_obj

def _multi_start_local_search(time_budget):
    """Run multiple local searches from different starting points."""
    rng = random.Random()
    best_A = None
    best_obj = float('inf')
    
    start_time = time.time()
    
    # Start with geometric progressions with various ratios
    for ratio in [1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5]:
        for start_val in [1, 2, 3, 5, 7, 10]:
            if time.time() - start_time > time_budget:
                break
            A = _generate_gp(ratio, start_val)
            if not _is_valid(A):
                continue
            remaining = time_budget - (time.time() - start_time)
            if remaining < 1:
                break
            A_sorted, obj = _local_search(A, remaining, rng)
            if obj < best_obj:
                best_obj = obj
                best_A = A_sorted
        if time.time() - start_time > time_budget:
            break
    
    # Try rational ratio geometric progressions
    rational_ratios = [(3, 2), (4, 3), (5, 4), (5, 3), (7, 5), (8, 5), (9, 7), (11, 8), (13, 10), (16, 13)]
    for p, q in rational_ratios:
        for start_val in [1, 2, 3, 5]:
            if time.time() - start_time > time_budget:
                break
            A = _generate_rational_gp(p, q, start_val)
            if not _is_valid(A):
                continue
            remaining = time_budget - (time.time() - start_time)
            if remaining < 1:
                break
            A_sorted, obj = _local_search(A, remaining, rng)
            if obj < best_obj:
                best_obj = obj
                best_A = A_sorted
        if time.time() - start_time > time_budget:
            break
    
    # Try random starts
    remaining = time_budget - (time.time() - start_time)
    if remaining > 2:
        max_starts = min(30, int(remaining / 2))
        for _ in range(max_starts):
            if time.time() - start_time > time_budget:
                break
            A = random.sample(range(1, 10001), 20)
            if not _is_valid(A):
                continue
            rem = time_budget - (time.time() - start_time)
            if rem < 1:
                break
            A_sorted, obj = _local_search(A, rem, rng)
            if obj < best_obj:
                best_obj = obj
                best_A = A_sorted
    
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
    
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = 80  # leave 10s safety margin
    
    try:
        best_A, best_obj = _multi_start_local_search(time_budget)
    except Exception:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
    
    # Final refinement with whatever time is left
    remaining = time_budget - (time.time() - start_time)
    if remaining > 1:
        try:
            refined_A, refined_obj = _local_search(best_A, remaining, random.Random())
            if refined_obj < best_obj:
                best_A = refined_A
        except Exception:
            pass
    
    # Ensure validity
    if not _is_valid(best_A):
        best_A = list(range(1, 21))
    
    return sorted(best_A)
