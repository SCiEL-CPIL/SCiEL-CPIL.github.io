import time
import random
import itertools
import numpy as np

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 85  # safety margin before 90s kill

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    A_arr = np.array(sorted(A), dtype=np.int64)
    n = len(A_arr)
    i_idx, j_idx = np.triu_indices(n)
    sums = A_arr[i_idx] + A_arr[j_idx]
    products = A_arr[i_idx] * A_arr[j_idx]
    return max(len(np.unique(sums)), len(np.unique(products)))

def _objective_fast(A):
    """Faster objective using Python sets for small n."""
    A_sorted = sorted(A)
    n = len(A_sorted)
    sums = set()
    products = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(A_sorted[i] + A_sorted[j])
            products.add(A_sorted[i] * A_sorted[j])
    return max(len(sums), len(products))

def _is_valid(A):
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > 10000:
            return False
    return True

def _smooth_numbers(limit, count):
    """Generate the 'count' smallest smooth numbers <= limit."""
    smooths = set()
    i_max = int(limit.bit_length())
    j_max = int(limit.bit_length())
    k_max = int(limit.bit_length())
    
    for i in range(i_max + 1):
        pow2 = 1 << i
        if pow2 > limit: break
        for j in range(j_max + 1):
            pow3 = pow2 * (3 ** j)
            if pow3 > limit: break
            for k in range(k_max + 1):
                val = pow3 * (5 ** k)
                if val > limit: break
                smooths.add(val)
    
    sorted_smooths = sorted(smooths)
    return sorted_smooths[:count]

def _construct_candidates():
    """Generate a list of candidate starting sets."""
    candidates = []
    
    # 1. Smooth numbers (powers of 2, 3, 5)
    try:
        smooths = _smooth_numbers(10000, 20)
        if len(smooths) == 20:
            candidates.append(smooths)
    except:
        pass
        
    # 2. Geometric Progressions
    for ratio in [1.2, 1.3, 1.5, 1.8, 2.0]:
        for start in [1, 2, 3]:
            A = []
            current = float(start)
            for _ in range(20):
                val = int(round(current))
                if val in A:
                    val += 1
                    while val in A:
                        val += 1
                if val > 10000:
                    break
                A.append(val)
                current *= ratio
            if len(A) == 20 and _is_valid(A):
                candidates.append(A)
                
    # 3. Arithmetic Progressions
    for start in [1, 2, 5]:
        for step in [1, 2, 3]:
            A = [start + i * step for i in range(20)]
            if max(A) <= 10000 and _is_valid(A):
                candidates.append(A)
                
    # 4. Random
    candidates.append(random.sample(range(1, 10001), 20))
    candidates.append(random.sample(range(1, 1001), 20))
    
    return candidates

def _local_search(A, time_limit, rng, use_double_move=False):
    """Local search with simulated annealing and improved move generation."""
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 30.0
    cooling_rate = 0.997
    
    while temperature > 0.05:
        if time.time() - start_time > time_limit:
            break
        
        if use_double_move and rng.random() < 0.3:
            # Double move: change two elements
            idx1 = rng.randint(0, 19)
            idx2 = rng.randint(0, 19)
            while idx2 == idx1:
                idx2 = rng.randint(0, 19)
            old_val1 = A[idx1]
            old_val2 = A[idx2]
            
            for _ in range(15):
                if time.time() - start_time > time_limit:
                    break
                
                r1 = rng.random()
                r2 = rng.random()
                if r1 < 0.5:
                    new_val1 = old_val1 + rng.randint(-10, 10)
                else:
                    new_val1 = rng.randint(1, 10000)
                
                if r2 < 0.5:
                    new_val2 = old_val2 + rng.randint(-10, 10)
                else:
                    new_val2 = rng.randint(1, 10000)
                
                if new_val1 < 1 or new_val1 > 10000 or new_val2 < 1 or new_val2 > 10000:
                    continue
                if new_val1 in A or new_val2 in A:
                    continue
                if new_val1 == new_val2:
                    continue
                
                A[idx1] = new_val1
                A[idx2] = new_val2
                new_obj = _objective(A)
                A[idx1] = old_val1
                A[idx2] = old_val2
                
                delta = new_obj - current_obj
                if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                    A[idx1] = new_val1
                    A[idx2] = new_val2
                    current_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_A = list(A)
                    break
        else:
            # Single move
            idx = rng.randint(0, 19)
            old_val = A[idx]
            accepted = False
            
            for _ in range(30):
                if time.time() - start_time > time_limit:
                    break
                
                r = rng.random()
                if r < 0.4:
                    new_val = old_val + rng.randint(-5, 5)
                elif r < 0.7:
                    new_val = old_val + rng.randint(-20, 20)
                elif r < 0.9:
                    new_val = rng.randint(1, max(2, old_val * 2))
                else:
                    new_val = rng.randint(1, 10000)
                
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in A:
                    continue
                
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
                    accepted = True
                    break
                    
        temperature *= cooling_rate
        
    return best_A, best_obj

def _polish_search(A, time_limit, rng):
    """High-precision polish with very tight temperature and exhaustive local moves."""
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 5.0
    cooling_rate = 0.999
    
    # Try all single-position moves with small perturbations
    while temperature > 0.01:
        if time.time() - start_time > time_limit:
            break
        
        idx = rng.randint(0, 19)
        old_val = A[idx]
        
        # Try a range of nearby values
        for delta in range(-15, 16):
            if time.time() - start_time > time_limit:
                break
            new_val = old_val + delta
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in A:
                continue
            
            A[idx] = new_val
            new_obj = _objective(A)
            A[idx] = old_val
            
            delta_obj = new_obj - current_obj
            if delta_obj <= 0 or rng.random() < min(1.0, max(0.0, -delta_obj / (temperature + 1e-10))):
                A[idx] = new_val
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
                break
        
        temperature *= cooling_rate
        
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5
    
    candidates = _construct_candidates()
    
    best_A = None
    best_obj = float('inf')
    
    rng = random.Random(42)
    
    # First pass: quick local search on each candidate
    for A in candidates:
        if time.time() - start_time > time_budget:
            break
        if not _is_valid(A):
            continue
        elapsed = time.time() - start_time
        remaining_for_this = max(1.0, time_budget - elapsed)
        refined_A, obj = _local_search(A, remaining_for_this, rng)
        if obj < best_obj:
            best_obj = obj
            best_A = refined_A
            
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
        
    # Second pass: more intensive refinement with double moves
    remaining = TIME_LIMIT - 5 - (time.time() - start_time)
    if remaining > 2:
        try:
            refined_A, refined_obj = _local_search(best_A, remaining * 0.5, rng, use_double_move=True)
            if refined_obj < best_obj:
                best_A = refined_A
                best_obj = refined_obj
        except Exception:
            pass
    
    # Final high-precision polish
    remaining = TIME_LIMIT - 5 - (time.time() - start_time)
    if remaining > 1:
        try:
            rng2 = random.Random(123)
            refined_A, refined_obj = _polish_search(best_A, remaining, rng2)
            if refined_obj < best_obj:
                best_A = refined_A
                best_obj = refined_obj
        except Exception:
            pass

    if not _is_valid(best_A):
        best_A = list(range(1, 21))
        
    return sorted(best_A)
