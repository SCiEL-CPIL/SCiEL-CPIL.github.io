import time
import random
import itertools
import numpy as np

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 85  # safety margin before 90s kill

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers.
    Uses NumPy vectorization for speed."""
    arr = np.array(sorted(A), dtype=np.int64)
    n = len(arr)
    # Upper-triangular indices
    i_idx, j_idx = np.triu_indices(n)
    sums = arr[i_idx] + arr[j_idx]
    prods = arr[i_idx] * arr[j_idx]
    # Count unique values
    n_sums = len(np.unique(sums))
    n_prods = len(np.unique(prods))
    return max(n_sums, n_prods)

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

def _geometric_start(start, ratio_num, ratio_den):
    """Generate a geometric progression with rational ratio p/q."""
    A = []
    val = start
    for _ in range(20):
        if val > MAX_VAL:
            break
        if val in A:
            while val in A:
                val += 1
            if val > MAX_VAL:
                break
        A.append(val)
        val = (val * ratio_num + ratio_den // 2) // ratio_den
    if len(A) == 20 and _is_valid(A):
        return A
    return None

def _logarithmic_set(base, count=20):
    """Generate numbers approximately at logarithmic spacing."""
    A = []
    for i in range(count):
        val = int(base * (1.3 ** i))
        if val > MAX_VAL:
            break
        if val not in A and val >= 1:
            A.append(val)
    while len(A) < count:
        A.append(len(A) + 1)
    if len(A) == 20 and _is_valid(A):
        return A
    return None

def _construct_candidates():
    """Generate a list of candidate starting sets."""
    candidates = []
    
    # 1. Smooth numbers
    try:
        smooths = _smooth_numbers(10000, 20)
        if len(smooths) == 20:
            candidates.append(smooths)
    except:
        pass
        
    # 2. Geometric Progressions
    rational_ratios = [
        (6, 5), (7, 6), (8, 7), (5, 4), (4, 3),
        (9, 8), (11, 10), (3, 2), (5, 3), (7, 5),
        (10, 9), (12, 11), (13, 12), (14, 13), (15, 14),
    ]
    for ratio_num, ratio_den in rational_ratios:
        for start in [1, 2, 3, 4, 5]:
            A = _geometric_start(start, ratio_num, ratio_den)
            if A is not None:
                candidates.append(A)
                
    # 3. Arithmetic Progressions
    for start in [1, 2, 5, 10]:
        for step in [1, 2, 3, 5]:
            A = [start + i * step for i in range(20)]
            if max(A) <= 10000 and _is_valid(A):
                candidates.append(A)
    
    # 4. Logarithmic sets
    for base in [1, 2, 3, 5]:
        A = _logarithmic_set(base)
        if A is not None:
            candidates.append(A)
    
    # 5. Random with different ranges
    rng = random.Random(123)
    for _ in range(3):
        candidates.append(rng.sample(range(1, 10001), 20))
    for _ in range(3):
        candidates.append(rng.sample(range(1, 1001), 20))
    for _ in range(2):
        candidates.append(rng.sample(range(1, 200), 20))
    
    return candidates

def _local_search(A, time_limit, rng):
    """Local search with simulated annealing, paired swaps, and collision-seeking moves."""
    A = sorted(A)
    A_set = set(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 30.0
    cooling_rate = 0.9992
    
    while temperature > 0.01:
        if time.time() - start_time > time_limit:
            break
        
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.35:
            # Single element change - small perturbation
            idx = rng.randint(0, 19)
            old_val = A[idx]
            
            r = rng.random()
            if r < 0.3:
                new_val = old_val + rng.randint(-3, 3)
            elif r < 0.6:
                new_val = old_val + rng.randint(-20, 20)
            elif r < 0.85:
                new_val = old_val + rng.randint(-100, 100)
            else:
                new_val = rng.randint(1, 10000)
            
            if new_val < 1 or new_val > 10000 or new_val in A_set:
                continue
            
            A[idx] = new_val
            A_set.discard(old_val)
            A_set.add(new_val)
            new_obj = _objective(A)
            
            delta = new_obj - current_obj
            if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
            else:
                # Revert
                A[idx] = old_val
                A_set.discard(new_val)
                A_set.add(old_val)
        
        elif move_type < 0.65:
            # Paired swap: change two elements simultaneously
            idx1, idx2 = rng.sample(range(20), 2)
            old1, old2 = A[idx1], A[idx2]
            
            new1 = old1 + rng.randint(-15, 15)
            new2 = old2 + rng.randint(-15, 15)
            
            if new1 < 1 or new1 > 10000 or new2 < 1 or new2 > 10000:
                continue
            if new1 == new2 or new1 in A_set or new2 in A_set:
                continue
            
            A[idx1] = new1
            A[idx2] = new2
            A_set.discard(old1)
            A_set.discard(old2)
            A_set.add(new1)
            A_set.add(new2)
            new_obj = _objective(A)
            
            delta = new_obj - current_obj
            if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
            else:
                # Revert
                A[idx1] = old1
                A[idx2] = old2
                A_set.discard(new1)
                A_set.discard(new2)
                A_set.add(old1)
                A_set.add(old2)
        
        else:
            # Collision-seeking: try to find a value that maximizes sum/product collisions
            idx = rng.randint(0, 19)
            old_val = A[idx]
            A_set.discard(old_val)
            
            # Generate candidate values that might create collisions
            candidates = set()
            # Values that would make ai + aj = ak + al for some k,l
            for j in range(20):
                if j == idx:
                    continue
                aj = A[j]
                # Try new_val such that new_val + aj hits an existing sum
                for k in range(20):
                    if k == idx or k == j:
                        continue
                    ak = A[k]
                    nv = ak - aj
                    if 1 <= nv <= 10000 and nv not in A_set:
                        candidates.add(nv)
                # Try new_val * aj hits an existing product
                for k in range(20):
                    if k == idx or k == j:
                        continue
                    ak = A[k]
                    if ak % aj == 0:
                        nv = ak // aj
                        if 1 <= nv <= 10000 and nv not in A_set:
                            candidates.add(nv)
            
            # Also add some random perturbations
            for _ in range(5):
                nv = old_val + rng.randint(-50, 50)
                if 1 <= nv <= 10000 and nv not in A_set:
                    candidates.add(nv)
            
            best_candidate = None
            best_candidate_obj = current_obj
            
            if candidates:
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                # Evaluate up to 15 candidates
                for nv in cand_list[:15]:
                    A[idx] = nv
                    obj = _objective(A)
                    if obj < best_candidate_obj:
                        best_candidate_obj = obj
                        best_candidate = nv
                A[idx] = old_val
            
            if best_candidate is not None:
                A[idx] = best_candidate
                A_set.add(best_candidate)
                current_obj = best_candidate_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_A = list(A)
            else:
                A_set.add(old_val)
        
        temperature *= cooling_rate
    
    # Final polish: try all single-element replacements with nearby values
    remaining = time_limit - (time.time() - start_time)
    if remaining > 1:
        A_polish = list(best_A)
        A_polish_set = set(A_polish)
        obj_polish = best_obj
        for i in range(20):
            if time.time() - start_time > time_limit - 0.5:
                break
            old = A_polish[i]
            for delta_val in range(-15, 16):
                if delta_val == 0:
                    continue
                nv = old + delta_val
                if nv < 1 or nv > 10000 or nv in A_polish_set:
                    continue
                A_polish[i] = nv
                A_polish_set.discard(old)
                A_polish_set.add(nv)
                o = _objective(A_polish)
                if o < obj_polish:
                    obj_polish = o
                    best_A = list(A_polish)
                    best_obj = obj_polish
                else:
                    A_polish[i] = old
                    A_polish_set.discard(nv)
                    A_polish_set.add(old)
    
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5
    
    candidates = _construct_candidates()
    
    best_A = None
    best_obj = float('inf')
    
    rng = random.Random(42)
    
    # Phase 1: Quick evaluation of all candidates
    candidate_objs = []
    for A in candidates:
        if not _is_valid(A):
            continue
        obj = _objective(A)
        candidate_objs.append((obj, A))
    
    candidate_objs.sort(key=lambda x: x[0])
    
    n_cands = len(candidate_objs)
    if n_cands == 0:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
    else:
        # Try top candidates with local search - give more time to top candidates
        top_n = min(10, n_cands)
        for i, (obj_init, A) in enumerate(candidate_objs[:top_n]):
            elapsed = time.time() - start_time
            remaining = time_budget - elapsed
            if remaining < 2:
                break
            # Allocate more time to better initial candidates
            if i < 3:
                alloc = min(remaining, 12.0)
            elif i < 6:
                alloc = min(remaining, 8.0)
            else:
                alloc = min(remaining, 5.0)
            refined_A, obj = _local_search(A, alloc, rng)
            if obj < best_obj:
                best_obj = obj
                best_A = refined_A
        
        # Try more candidates if time permits
        for obj_init, A in candidate_objs[top_n:]:
            elapsed = time.time() - start_time
            remaining = time_budget - elapsed
            if remaining < 3:
                break
            if obj_init >= best_obj:
                continue
            refined_A, obj = _local_search(A, min(remaining, 4.0), rng)
            if obj < best_obj:
                best_obj = obj
                best_A = refined_A
    
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
        
    # Final refinement with multiple restarts
    remaining = TIME_LIMIT - 5 - (time.time() - start_time)
    if remaining > 3:
        try:
            # Do a final local search from the best solution
            refined_A, refined_obj = _local_search(best_A, remaining, rng)
            if refined_obj < best_obj:
                best_A = refined_A
                best_obj = refined_obj
        except Exception:
            pass

    if not _is_valid(best_A):
        best_A = list(range(1, 21))
        
    return sorted(best_A)
