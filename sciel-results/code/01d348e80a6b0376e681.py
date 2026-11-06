import time
import random
import itertools
import numpy as np

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 85  # safety margin before 90s kill

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers using NumPy."""
    arr = np.array(sorted(A), dtype=np.int64)
    n = len(arr)
    
    # Sumset: all pairs (i, j) with i <= j
    i_idx = np.repeat(np.arange(n), n)
    j_idx = np.tile(np.arange(n), n)
    # Only keep i <= j to avoid duplicates (but we need all i,j for sums since a+b=b+a)
    # Actually for sumset, (i,j) and (j,i) give same sum, so we can just do all pairs
    # But to be safe and match original: use i <= j
    mask = i_idx <= j_idx
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    
    sums = arr[i_idx] + arr[j_idx]
    n_sums = len(np.unique(sums))
    
    # Product set: all pairs (i, j) with i <= j
    products = arr[i_idx] * arr[j_idx]
    n_products = len(np.unique(products))
    
    return max(n_sums, n_products)

def _objective_fast(A):
    """Faster objective using sorted arrays and vectorized unique."""
    arr = np.array(sorted(A), dtype=np.int64)
    n = len(arr)
    
    # For sumset: all pairs i <= j
    # Use broadcasting
    sums = arr[:, None] + arr[None, :]
    # Only upper triangle (including diagonal)
    tri_idx = np.triu_indices(n)
    sums_tri = sums[tri_idx]
    n_sums = len(np.unique(sums_tri))
    
    # For product set
    products = arr[:, None] * arr[None, :]
    products_tri = products[tri_idx]
    n_products = len(np.unique(products_tri))
    
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

def _generate_two_gp(start_time, time_budget):
    """Generate sets as union of two geometric progressions (e.g., powers of 2 and 3)."""
    best_A = None
    best_obj = float('inf')
    
    # Try various combinations of two bases
    bases = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30]
    
    for b1 in bases:
        for b2 in bases:
            if b1 >= b2:
                continue
            if time.time() - start_time > time_budget:
                return best_A, best_obj
            
            # Generate all possible values: a1 * b1^i and a2 * b2^j
            # We need to pick 20 distinct values from the union
            candidates = set()
            
            # Powers of b1 with various starting values
            for start1 in range(1, min(20, MAX_VAL // (b1 ** 10)) + 1):
                val = start1
                for _ in range(15):
                    if val <= MAX_VAL:
                        candidates.add(val)
                    else:
                        break
                    val *= b1
            
            # Powers of b2 with various starting values
            for start2 in range(1, min(20, MAX_VAL // (b2 ** 10)) + 1):
                val = start2
                for _ in range(15):
                    if val <= MAX_VAL:
                        candidates.add(val)
                    else:
                        break
                    val *= b2
            
            if len(candidates) < 20:
                continue
            
            # Try to pick 20 values that minimize the objective
            # Use a greedy approach: start with the smallest values, then try swaps
            cand_list = sorted(candidates)
            
            # Initial selection: take first 20
            A = cand_list[:20]
            obj = _objective_fast(A)
            if obj < best_obj:
                best_obj = obj
                best_A = list(A)
            
            # Try some random selections from candidates
            for _ in range(10):
                if time.time() - start_time > time_budget:
                    return best_A, best_obj
                A = random.sample(cand_list, 20)
                A.sort()
                obj = _objective_fast(A)
                if obj < best_obj:
                    best_obj = obj
                    best_A = list(A)
    
    return best_A, best_obj

def _local_search(A, time_limit, rng):
    """Local search with simulated annealing to minimize the objective."""
    A = sorted(A)
    current_obj = _objective_fast(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 50.0
    cooling_rate = 0.998
    min_temperature = 0.1
    
    while temperature > min_temperature:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random element to modify
        idx = rng.randint(0, 19)
        old_val = A[idx]
        
        # Try a few random changes
        for _ in range(30):
            if time.time() - start_time > time_limit:
                break
            
            # Random delta - bias toward small changes
            if rng.random() < 0.4:
                new_val = old_val + rng.randint(-20, 20)
            elif rng.random() < 0.7:
                new_val = old_val + rng.randint(-100, 100)
            else:
                new_val = old_val + rng.randint(-500, 500)
            
            if new_val < 1 or new_val > MAX_VAL:
                continue
            if new_val in A:
                continue
            
            # Test the change
            A[idx] = new_val
            new_obj = _objective_fast(A)
            A[idx] = old_val
            
            delta = new_obj - current_obj
            if delta <= 0 or (temperature > 0.5 and rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10)))):
                A[idx] = new_val
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
                break
        
        temperature *= cooling_rate
    
    return best_A, best_obj

def _swap_local_search(A, time_limit, rng):
    """Local search that tries swapping elements between two geometric clusters."""
    A = sorted(A)
    current_obj = _objective_fast(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    
    # Identify the two "clusters" - split into lower and upper halves
    mid = 10
    
    while time.time() - start_time < time_limit:
        # Try swapping an element from lower cluster with one from upper
        i = rng.randint(0, mid - 1)
        j = rng.randint(mid, 19)
        
        # Generate candidate values for swap
        for _ in range(20):
            if time.time() - start_time > time_limit:
                return best_A, best_obj
            
            # Try to replace A[i] with a value near A[j] (move up)
            # or replace A[j] with a value near A[i] (move down)
            if rng.random() < 0.5:
                # Replace lower element with something higher
                new_val = A[j] + rng.randint(-10, 10)
                if 1 <= new_val <= MAX_VAL and new_val not in A:
                    old = A[i]
                    A[i] = new_val
                    new_obj = _objective_fast(A)
                    if new_obj <= current_obj:
                        current_obj = new_obj
                        if new_obj < best_obj:
                            best_obj = new_obj
                            best_A = list(A)
                    else:
                        A[i] = old
            else:
                # Replace upper element with something lower
                new_val = A[i] + rng.randint(-10, 10)
                if 1 <= new_val <= MAX_VAL and new_val not in A:
                    old = A[j]
                    A[j] = new_val
                    new_obj = _objective_fast(A)
                    if new_obj <= current_obj:
                        current_obj = new_obj
                        if new_obj < best_obj:
                            best_obj = new_obj
                            best_A = list(A)
                    else:
                        A[j] = old
    
    return best_A, best_obj

def _geometric_start(ratio, start_val):
    """Generate a geometric-like starting set."""
    A = set()
    current = start_val
    while len(A) < 20 and current <= MAX_VAL:
        A.add(current)
        current = int(current * ratio)
        if current <= start_val:
            break
    # Fill remaining with small random values if needed
    if len(A) < 20:
        while len(A) < 20:
            v = random.randint(1, MAX_VAL)
            A.add(v)
    return list(A)

def _random_start():
    """Generate a random valid starting set."""
    while True:
        A = random.sample(range(1, MAX_VAL + 1), 20)
        if _is_valid(A):
            return A

def _multi_start_local_search(time_budget):
    """Run multiple local searches from different starting points."""
    rng = random.Random()
    best_A = None
    best_obj = float('inf')
    
    start_time = time.time()
    
    # First, try the two-GP strategy
    gp_A, gp_obj = _generate_two_gp(start_time, min(20, time_budget * 0.3))
    if gp_A is not None and gp_obj < best_obj:
        best_obj = gp_obj
        best_A = gp_A
    
    # Start with geometric progressions
    remaining = time_budget - (time.time() - start_time)
    if remaining > 5:
        for ratio in [1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5]:
            if time.time() - start_time > time_budget:
                break
            for start_val in [1, 2, 3, 5, 7, 10]:
                if time.time() - start_time > time_budget:
                    break
                A = _geometric_start(ratio, start_val)
                if not _is_valid(A):
                    continue
                A_sorted, obj = _local_search(A, max(1, time_budget - (time.time() - start_time)), rng)
                if obj < best_obj:
                    best_obj = obj
                    best_A = A_sorted
    
    # Try random starts
    remaining = time_budget - (time.time() - start_time)
    if remaining > 3:
        for _ in range(15):
            if time.time() - start_time > time_budget:
                break
            A = _random_start()
            A_sorted, obj = _local_search(A, max(1, time_budget - (time.time() - start_time)), rng)
            if obj < best_obj:
                best_obj = obj
                best_A = A_sorted
    
    # Swap-based local search on best result
    if best_A is not None:
        remaining = time_budget - (time.time() - start_time)
        if remaining > 2:
            swap_A, swap_obj = _swap_local_search(best_A, remaining, rng)
            if swap_obj < best_obj:
                best_obj = swap_obj
                best_A = swap_A
    
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective_fast(best_A)
    
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5  # leave margin
    
    try:
        best_A, best_obj = _multi_start_local_search(time_budget)
    except Exception:
        best_A = list(range(1, 21))
        best_obj = _objective_fast(best_A)
    
    # Final refinement with whatever time is left
    remaining = TIME_LIMIT - 5 - (time.time() - start_time)
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

# EVOLVE-BLOCK-END
