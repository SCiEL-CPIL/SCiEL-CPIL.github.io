import time
import random
import math

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 85.0

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    A_sorted = sorted(A)
    n = len(A_sorted)
    
    # Sumset with equal summands allowed
    sums = set()
    for i in range(n):
        ai = A_sorted[i]
        for j in range(i, n):
            sums.add(ai + A_sorted[j])
    
    # Product set with equal factors allowed
    products = set()
    for i in range(n):
        ai = A_sorted[i]
        for j in range(i, n):
            products.add(ai * A_sorted[j])
    
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

def _generate_log_spaced(start_val, end_val, n_points=20):
    """Generate log-spaced integers between start_val and end_val."""
    if start_val < 1:
        start_val = 1
    if end_val > MAX_VAL:
        end_val = MAX_VAL
    if start_val >= end_val:
        return list(range(1, min(20, MAX_VAL) + 1))
    
    A = set()
    log_start = math.log(start_val)
    log_end = math.log(end_val)
    
    for i in range(n_points):
        log_val = log_start + (log_end - log_start) * i / (n_points - 1)
        val = int(round(math.exp(log_val)))
        val = max(1, min(MAX_VAL, val))
        A.add(val)
    
    # Fill gaps if needed
    if len(A) < n_points:
        # Try to add values near existing ones
        existing = sorted(A)
        candidates = []
        for i in range(len(existing) - 1):
            mid = (existing[i] + existing[i+1]) // 2
            if mid not in A and 1 <= mid <= MAX_VAL:
                candidates.append(mid)
        random.shuffle(candidates)
        for v in candidates:
            if len(A) >= n_points:
                break
            if v not in A:
                A.add(v)
        
        # Still need more? Add random values
        if len(A) < n_points:
            for _ in range(100):
                v = random.randint(1, MAX_VAL)
                if v not in A:
                    A.add(v)
                    if len(A) >= n_points:
                        break
    
    return list(A)

def _generate_gp(ratio, start_val):
    """Generate a geometric progression set, filling gaps if needed."""
    A = set()
    current = start_val
    while len(A) < 20 and current <= MAX_VAL:
        A.add(current)
        current = int(current * ratio)
        if current <= start_val:
            break
    
    if len(A) < 20:
        # Fill remaining with log-spaced values
        if A:
            min_val = min(A)
            max_val = max(A)
            if max_val < MAX_VAL:
                extra = _generate_log_spaced(max_val + 1, MAX_VAL, 20 - len(A))
                for v in extra:
                    if len(A) >= 20:
                        break
                    if v not in A:
                        A.add(v)
        
        if len(A) < 20:
            for v in range(1, MAX_VAL + 1):
                if len(A) >= 20:
                    break
                if v not in A:
                    A.add(v)
    return list(A)

def _generate_ap(start_val, step):
    """Generate an arithmetic progression."""
    A = set()
    current = start_val
    while len(A) < 20 and current <= MAX_VAL:
        A.add(current)
        current += step
    if len(A) < 20:
        for v in range(1, MAX_VAL + 1):
            if len(A) >= 20:
                break
            if v not in A:
                A.add(v)
    return list(A)

def _local_search(A, time_limit, rng, mode='sa'):
    """Local search with simulated annealing and improved move generation."""
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    
    if mode == 'sa':
        temperature = 5.0
        cooling_rate = 0.995
        min_temperature = 0.01
    else:
        temperature = 0.0
        cooling_rate = 1.0
        min_temperature = 0.0

    A_set = set(A)
    
    while temperature > min_temperature:
        if time.time() - start_time > time_limit:
            break
        
        idx = rng.randint(0, 19)
        old_val = A[idx]
        
        # Improved move generation: try multiple strategies
        new_val = None
        strategy = rng.random()
        
        if strategy < 0.3:
            # Small perturbation around current value
            step = rng.randint(-5, 5)
            new_val = old_val + step
        elif strategy < 0.5:
            # Try values near other elements in the set
            other_idx = rng.randint(0, 19)
            if other_idx != idx:
                other_val = A[other_idx]
                offset = rng.randint(-3, 3)
                new_val = other_val + offset
        elif strategy < 0.7:
            # Log-spaced suggestion: try values at geometric positions
            if old_val > 1:
                log_old = math.log(old_val)
                new_log = log_old + rng.uniform(-0.3, 0.3)
                new_val = int(round(math.exp(new_log)))
            else:
                new_val = rng.randint(1, 10)
        elif strategy < 0.85:
            # Random value in a reasonable range
            low = max(1, old_val // 3)
            high = min(MAX_VAL, old_val * 3)
            new_val = rng.randint(low, high)
        else:
            # Completely random
            new_val = rng.randint(1, MAX_VAL)
        
        if new_val is None or new_val < 1 or new_val > MAX_VAL:
            continue
        if new_val in A_set:
            continue
        
        # Apply move
        A[idx] = new_val
        A_set.remove(old_val)
        A_set.add(new_val)
        new_obj = _objective(A)
        
        delta = new_obj - current_obj
        
        if delta <= 0 or rng.random() < math.exp(-delta / (temperature + 1e-10)):
            current_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_A = list(A)
        else:
            A[idx] = old_val
            A_set.remove(new_val)
            A_set.add(old_val)
            
        temperature *= cooling_rate
        
    return best_A, best_obj

def _multi_start_search(time_budget):
    rng = random.Random(42)
    best_A = None
    best_obj = float('inf')
    
    start_time = time.time()
    
    # Generate diverse starting points
    starts = []
    
    # 1. Log-spaced sets with different ranges
    for s, e in [(1, 100), (1, 500), (1, 2000), (1, 10000), (2, 100), (5, 500)]:
        starts.append(_generate_log_spaced(s, e))
    
    # 2. Geometric progressions
    for r in [1.2, 1.4, 1.6, 1.8, 2.0, 2.5]:
        for s in [1, 2, 3]:
            starts.append(_generate_gp(r, s))
    
    # 3. Arithmetic progressions
    for step in [1, 2, 3, 5, 10]:
        starts.append(_generate_ap(1, step))
    
    # 4. Random sets from small range
    for _ in range(5):
        starts.append(random.sample(range(1, 50), 20))
    
    # 5. Random sets from medium range
    for _ in range(5):
        starts.append(random.sample(range(1, 500), 20))
    
    # Shuffle starts for diversity
    rng.shuffle(starts)
    
    # Allocate time to each start
    num_starts = len(starts)
    per_start_time = time_budget / num_starts
    
    for i, A in enumerate(starts):
        if time.time() - start_time > time_budget:
            break
        
        if not _is_valid(A):
            continue
        
        remaining = time_budget - (time.time() - start_time)
        slice_time = min(per_start_time, remaining * 0.9)
        
        if slice_time < 0.3:
            continue
        
        A_sol, obj = _local_search(A, slice_time, rng)
        if obj < best_obj:
            best_obj = obj
            best_A = A_sol
    
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
        
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 2.0  # Safety margin
    
    try:
        best_A, best_obj = _multi_start_search(time_budget)
    except Exception:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
        
    # Final refinement with remaining time
    remaining = TIME_LIMIT - 2.0 - (time.time() - start_time)
    if remaining > 1:
        try:
            refined_A, refined_obj = _local_search(best_A, remaining, random.Random(99))
            if refined_obj < best_obj:
                best_A = refined_A
        except Exception:
            pass
            
    if not _is_valid(best_A):
        best_A = list(range(1, 21))
        
    return sorted(best_A)

# EVOLVE-BLOCK-END
