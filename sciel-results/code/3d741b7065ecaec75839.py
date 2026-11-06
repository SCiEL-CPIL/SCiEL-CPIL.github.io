import time
import random
import itertools
import numpy as np

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 85  # safety margin before 90s kill

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers.
    Uses bitset for sums (bounded) and sorted-unique for products."""
    arr = np.array(sorted(A), dtype=np.int64)
    n = len(arr)
    # Upper-triangular indices including diagonal
    i_idx, j_idx = np.triu_indices(n)
    sums = arr[i_idx] + arr[j_idx]
    prods = arr[i_idx] * arr[j_idx]
    # Sums are bounded: max sum <= 20000, use bitset
    max_sum = int(sums.max())
    sum_bitset = np.zeros(max_sum + 1, dtype=np.uint8)
    sum_bitset[sums] = 1
    n_sums = sum_bitset.sum()
    # Products can be up to 10^8, use sorted unique
    prods_sorted = np.sort(prods)
    n_prods = 1 + int(np.sum(prods_sorted[1:] != prods_sorted[:-1]))
    return max(int(n_sums), n_prods)

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

def _dual_geometric_set(start1, ratio1_num, ratio1_den, n1, start2, ratio2_num, ratio2_den, n2):
    """Generate a set that is a union of two geometric progressions.
    This maximizes multiplicative collisions: products within each GP collide
    due to shared prime factors, and cross-products may also collide."""
    A = []
    # First GP
    val = start1
    for _ in range(n1):
        if val > MAX_VAL:
            break
        if val not in A and val >= 1:
            A.append(val)
        val = (val * ratio1_num + ratio1_den // 2) // ratio1_den
    # Second GP
    val = start2
    for _ in range(n2):
        if val > MAX_VAL:
            break
        if val not in A and val >= 1:
            A.append(val)
        val = (val * ratio2_num + ratio2_den // 2) // ratio2_den
    # Fill remaining if needed
    while len(A) < 20:
        for v in range(1, MAX_VAL + 1):
            if v not in A:
                A.append(v)
                break
        if len(A) < 20:
            break
    if len(A) == 20 and _is_valid(A):
        return A
    return None

def _powers_of_two_and_three(n2, n3):
    """Generate set of powers of 2 and powers of 3, choosing n2+n3=20 total."""
    A = []
    # Powers of 2
    v = 1
    for _ in range(n2):
        if v > MAX_VAL:
            break
        if v not in A:
            A.append(v)
        v *= 2
    # Powers of 3
    v = 1
    for _ in range(n3):
        if v > MAX_VAL:
            break
        if v not in A:
            A.append(v)
        v *= 3
    # Fill if needed
    while len(A) < 20:
        for v in range(1, MAX_VAL + 1):
            if v not in A:
                A.append(v)
                break
        if len(A) < 20:
            break
    if len(A) == 20 and _is_valid(A):
        return A
    return None

def _truncated_gp(a, p, q, n=20):
    """Generate truncated geometric progression: a*(p/q)^i rounded to nearest int.
    Returns list of n distinct positive integers <= 10000, or None if invalid."""
    A = []
    val = a
    for i in range(n):
        v = int(round(val))
        if v < 1 or v > 10000:
            break
        if v not in A:
            A.append(v)
        else:
            # Try small adjustment
            for delta in range(1, 10):
                for dv in [v + delta, v - delta]:
                    if 1 <= dv <= 10000 and dv not in A:
                        A.append(dv)
                        break
                else:
                    continue
                break
            else:
                break
        val = val * p / q
    if len(A) == n and _is_valid(A):
        return sorted(A)
    return None

def _construct_candidates():
    """Generate a list of candidate starting sets."""
    candidates = []
    seen = set()
    
    def _add(A):
        if A is not None and _is_valid(A):
            key = tuple(sorted(A))
            if key not in seen:
                seen.add(key)
                candidates.append(A)
    
    # 1. Smooth numbers
    try:
        smooths = _smooth_numbers(10000, 20)
        if len(smooths) == 20:
            _add(smooths)
    except:
        pass
        
    # 2. Dual geometric sets (key strategy: two GPs for multiplicative collisions)
    for n2 in range(5, 16):
        n3 = 20 - n2
        _add(_powers_of_two_and_three(n2, n3))
    
    dual_configs = [
        (1, 2, 1, 10, 3, 3, 1, 10),
        (1, 2, 1, 12, 5, 5, 1, 8),
        (1, 3, 1, 8, 2, 2, 1, 12),
        (2, 2, 1, 10, 3, 3, 1, 10),
        (1, 4, 1, 8, 3, 3, 1, 12),
        (1, 2, 1, 10, 9, 9, 1, 10),
        (1, 3, 1, 10, 4, 4, 1, 10),
        (1, 2, 1, 15, 5, 5, 1, 5),
        (1, 5, 1, 8, 2, 2, 1, 12),
        (1, 2, 1, 10, 7, 7, 1, 10),
        (1, 2, 1, 13, 3, 3, 1, 7),
        (1, 3, 1, 13, 2, 2, 1, 7),
        (1, 2, 1, 10, 4, 4, 1, 10),
        (1, 4, 1, 10, 2, 2, 1, 10),
    ]
    for cfg in dual_configs:
        _add(_dual_geometric_set(*cfg))
    
    # 3. Deterministic GP grid search over rational ratios p/q
    # Key insight: GP with ratio close to 1 gives compact sumset;
    # GP with small prime ratio gives compact product set
    ratio_list = [
        # (p, q) - ratios slightly above 1 for compact sums
        (5, 4), (6, 5), (7, 6), (8, 7), (9, 8), (10, 9), (11, 10), (12, 11), (13, 12), (14, 13), (15, 14), (16, 15), (17, 16), (18, 17), (19, 18), (20, 19),
        # Integer ratios for multiplicative structure
        (2, 1), (3, 1), (4, 1), (5, 1),
        # Other useful ratios
        (3, 2), (4, 3), (5, 3), (7, 5), (5, 4), (9, 7), (11, 8),
        (7, 4), (8, 5), (9, 5), (10, 7), (11, 7), (12, 7), (13, 8),
        (14, 9), (15, 10), (16, 11), (17, 12), (18, 13), (19, 14),
    ]
    for p, q in ratio_list:
        for a in range(1, 30):
            A = _truncated_gp(a, p, q, 20)
            if A is not None:
                _add(A)
    
    # Also try with larger starting values for ratios > 1
    for p, q in [(5, 4), (6, 5), (7, 6), (3, 2), (4, 3)]:
        for a in range(10, 100, 5):
            A = _truncated_gp(a, p, q, 20)
            if A is not None:
                _add(A)
    
    # 4. Arithmetic Progressions
    for start in [1, 2, 5, 10, 15, 20]:
        for step in [1, 2, 3, 4, 5, 7, 10]:
            A = [start + i * step for i in range(20)]
            if max(A) <= 10000:
                _add(A)
    
    # 5. Logarithmic sets
    for base in [1, 2, 3, 5, 7, 10]:
        _add(_logarithmic_set(base))
    
    # 6. Random with different ranges
    rng = random.Random(123)
    for _ in range(3):
        _add(rng.sample(range(1, 10001), 20))
    for _ in range(3):
        _add(rng.sample(range(1, 1001), 20))
    for _ in range(2):
        _add(rng.sample(range(1, 200), 20))
    
    return candidates

def _local_search(A, time_limit, rng):
    """Local search with simulated annealing, paired swaps, and GP-aware moves."""
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 30.0
    cooling_rate = 0.999  # slower cooling for more iterations
    
    # Precompute set for O(1) membership
    A_set = set(A)
    
    while temperature > 0.01:
        if time.time() - start_time > time_limit:
            break
        
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.30:
            # Single element change
            idx = rng.randint(0, 19)
            old_val = A[idx]
            
            r = rng.random()
            if r < 0.25:
                # Small perturbation
                new_val = old_val + rng.randint(-5, 5)
            elif r < 0.45:
                # Medium perturbation
                new_val = old_val + rng.randint(-50, 50)
            elif r < 0.65:
                # Try to create multiplicative collision
                existing = rng.choice(A)
                if rng.random() < 0.5:
                    k = rng.randint(2, 5)
                    new_val = existing * k
                else:
                    k = rng.randint(2, 5)
                    new_val = existing // k
                    if new_val < 1:
                        new_val = existing * k
            elif r < 0.80:
                # Random
                new_val = rng.randint(1, 10000)
            elif r < 0.90:
                # Try to hit a power of 2 or 3
                if rng.random() < 0.5:
                    p = rng.randint(0, 13)
                    new_val = 2 ** p
                else:
                    p = rng.randint(0, 8)
                    new_val = 3 ** p
            else:
                # GP-aware: try to place element at a*(p/q)^k position
                # relative to existing structure
                base = A[0]
                p, q = rng.choice([(5,4),(6,5),(7,6),(3,2),(4,3),(2,1)])
                k = rng.randint(0, 15)
                new_val = int(round(base * (p/q)**k))
            
            if new_val < 1 or new_val > 10000 or new_val in A_set:
                continue
            
            A[idx] = new_val
            A_set.discard(old_val)
            A_set.add(new_val)
            new_obj = _objective(A)
            A[idx] = old_val
            A_set.discard(new_val)
            A_set.add(old_val)
            
            delta = new_obj - current_obj
            if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                A[idx] = new_val
                A_set.discard(old_val)
                A_set.add(new_val)
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
        
        elif move_type < 0.55:
            # Paired swap: change two elements simultaneously
            idx1, idx2 = rng.sample(range(20), 2)
            old1, old2 = A[idx1], A[idx2]
            
            new1 = old1 + rng.randint(-20, 20)
            new2 = old2 + rng.randint(-20, 20)
            
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
            A[idx1] = old1
            A[idx2] = old2
            A_set.discard(new1)
            A_set.discard(new2)
            A_set.add(old1)
            A_set.add(old2)
            
            delta = new_obj - current_obj
            if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                A[idx1] = new1
                A[idx2] = new2
                A_set.discard(old1)
                A_set.discard(old2)
                A_set.add(new1)
                A_set.add(new2)
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
        
        elif move_type < 0.80:
            # GP-aware move: try to make two elements form a geometric pair
            idx1, idx2 = rng.sample(range(20), 2)
            old1, old2 = A[idx1], A[idx2]
            
            k = rng.choice([2, 3, 4, 5])
            if rng.random() < 0.5:
                new1 = old1
                new2 = old1 * k
            else:
                new2 = old2
                new1 = old2 * k
            
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
            A[idx1] = old1
            A[idx2] = old2
            A_set.discard(new1)
            A_set.discard(new2)
            A_set.add(old1)
            A_set.add(old2)
            
            delta = new_obj - current_obj
            if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                A[idx1] = new1
                A[idx2] = new2
                A_set.discard(old1)
                A_set.discard(old2)
                A_set.add(new1)
                A_set.add(new2)
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
        
        else:
            # Multi-element GP injection: try to replace 3-4 elements with a GP segment
            n_replace = rng.choice([3, 4])
            indices = rng.sample(range(20), n_replace)
            old_vals = [A[i] for i in indices]
            
            # Generate a small GP
            base = rng.randint(1, 50)
            p, q = rng.choice([(5,4),(6,5),(7,6),(3,2),(4,3),(2,1),(3,1)])
            new_vals = []
            val = base
            for _ in range(n_replace):
                v = int(round(val))
                if 1 <= v <= 10000 and v not in A_set:
                    new_vals.append(v)
                else:
                    new_vals = None
                    break
                val = val * p / q
            
            if new_vals is None or len(new_vals) != n_replace:
                continue
            
            # Check all new values are distinct and not in A
            if len(set(new_vals)) != n_replace:
                continue
            for nv in new_vals:
                if nv in A_set:
                    break
            else:
                # Apply
                for i, idx in enumerate(indices):
                    A[idx] = new_vals[i]
                for ov in old_vals:
                    A_set.discard(ov)
                for nv in new_vals:
                    A_set.add(nv)
                
                new_obj = _objective(A)
                
                # Revert
                for i, idx in enumerate(indices):
                    A[idx] = old_vals[i]
                for nv in new_vals:
                    A_set.discard(nv)
                for ov in old_vals:
                    A_set.add(ov)
                
                delta = new_obj - current_obj
                if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                    for i, idx in enumerate(indices):
                        A[idx] = new_vals[i]
                    for ov in old_vals:
                        A_set.discard(ov)
                    for nv in new_vals:
                        A_set.add(nv)
                    current_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_A = list(A)
        
        temperature *= cooling_rate
    
    # Final polish: try all single-element replacements with nearby values
    if time.time() - start_time < time_limit - 1:
        A_polish = list(best_A)
        obj_polish = best_obj
        A_set_p = set(A_polish)
        for i in range(20):
            if time.time() - start_time > time_limit - 1:
                break
            old = A_polish[i]
            # Try nearby values and values that create multiplicative structure
            candidates_nv = list(range(max(1, old - 20), min(10001, old + 21)))
            for k in [2, 3, 4, 5, 6]:
                if old * k <= 10000:
                    candidates_nv.append(old * k)
                if old % k == 0 and old // k >= 1:
                    candidates_nv.append(old // k)
            # Also try GP positions relative to other elements
            for other in A_polish:
                for p, q in [(5,4),(6,5),(3,2)]:
                    for k in range(0, 5):
                        v = int(round(other * (p/q)**k))
                        if 1 <= v <= 10000:
                            candidates_nv.append(v)
            
            for nv in set(candidates_nv):
                if nv == old or nv < 1 or nv > 10000 or nv in A_set_p:
                    continue
                A_polish[i] = nv
                A_set_p.discard(old)
                A_set_p.add(nv)
                o = _objective(A_polish)
                if o < obj_polish:
                    obj_polish = o
                    best_A = list(A_polish)
                    best_obj = obj_polish
                else:
                    A_polish[i] = old
                    A_set_p.discard(nv)
                    A_set_p.add(old)
    
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
        # Try top candidates with local search
        # Give more time to top candidates
        top_n = min(12, n_cands)
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
            elif i < 9:
                alloc = min(remaining, 5.0)
            else:
                alloc = min(remaining, 4.0)
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
        
    # Final refinement with remaining time
    remaining = TIME_LIMIT - 5 - (time.time() - start_time)
    if remaining > 2:
        try:
            refined_A, refined_obj = _local_search(best_A, remaining, rng)
            if refined_obj < best_obj:
                best_A = refined_A
        except Exception:
            pass

    if not _is_valid(best_A):
        best_A = list(range(1, 21))
        
    return sorted(best_A)
