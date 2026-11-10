import time
import random
import numpy as np

def _calc_score(A):
    """Compute max(|A+A|, |A*A|) for set A using vectorized operations."""
    A = np.array(list(A), dtype=np.int64)
    n = len(A)
    i_idx, j_idx = np.triu_indices(n)
    sums = A[i_idx] + A[j_idx]
    prods = A[i_idx] * A[j_idx]
    sumset_size = len(np.unique(sums))
    prodset_size = len(np.unique(prods))
    return max(sumset_size, prodset_size)

def _generate_smooth_pool(n_max=10000):
    """Generate a diverse pool of smooth numbers and small integers."""
    pool = set()
    # Small integers
    pool.update(range(1, 51))
    # 2-smooth
    for a in range(14):
        val = 1 << a
        while val <= n_max:
            pool.add(val)
            val *= 3
    # 3-smooth  
    for a in range(14):
        for b in range(9):
            val = (1 << a) * (3 ** b)
            if val <= n_max:
                pool.add(val)
                # 4-smooth
                for c in range(6):
                    val2 = val * (5 ** c)
                    if val2 <= n_max:
                        pool.add(val2)
    return sorted(pool)

def _local_search(A, candidate_pool, max_iters=8000, timeout=30.0):
    """Aggressive local search with multiple swap strategies."""
    A = sorted(set(A))
    if len(A) != 20:
        return A, float('inf')
    
    current_score = _calc_score(A)
    best_A = list(A)
    best_score = current_score
    start_time = time.time()
    A_set = set(A)
    
    for it in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        # Strategy 1: Single swap
        idx = random.randint(0, 19)
        candidates = [x for x in candidate_pool if x not in A_set]
        if not candidates:
            break
        
        new_elem = random.choice(candidates)
        new_A = A.copy()
        new_A[idx] = new_elem
        new_A = sorted(set(new_A))
        
        if len(new_A) == 20:
            new_score = _calc_score(new_A)
            if new_score < best_score:
                best_A = new_A
                best_score = new_score
                A = new_A
                A_set = set(A)
    
    return best_A, best_score

def _polish_pass(A, timeout=10.0):
    """Final high-precision polish with small perturbations."""
    A = sorted(set(A))
    if len(A) != 20:
        return A, float('inf')
    
    current_score = _calc_score(A)
    best_A = list(A)
    best_score = current_score
    start_time = time.time()
    A_set = set(A)
    
    for it in range(50000):
        if time.time() - start_time > timeout:
            break
        
        idx = random.randint(0, 19)
        old_val = A[idx]
        
        # Try very small perturbations
        for delta in range(1, 10):
            for new_val in [old_val - delta, old_val + delta]:
                if new_val < 1 or new_val > 10000 or new_val in A_set:
                    continue
                
                new_A = A.copy()
                new_A[idx] = new_val
                new_A = sorted(set(new_A))
                
                if len(new_A) != 20:
                    continue
                
                new_score = _calc_score(new_A)
                if new_score < best_score:
                    best_A = new_A
                    best_score = new_score
                    A = new_A
                    A_set = set(A)
                    break
            else:
                continue
            break
    
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Generate candidate pool
    candidate_pool = _generate_smooth_pool(10000)
    random.shuffle(candidate_pool)
    
    # Phase 1: Evaluate multiple initial candidates, prioritizing AP
    initial_sets = []
    
    # 1. Arithmetic progression (minimizes sumset to 39)
    initial_sets.append(list(range(1, 21)))
    
    # 2. Scaled AP variants
    for scale in [2, 3, 5]:
        ap = [scale * i for i in range(1, 21)]
        if max(ap) <= 10000:
            initial_sets.append(ap)
    
    # 3. Small smooth numbers
    small_smooth = [x for x in candidate_pool if x <= 100][:20]
    if len(small_smooth) >= 20:
        initial_sets.append(small_smooth)
    
    # 4. Geometric progression
    geo = []
    val = 1
    for i in range(20):
        val = int(round(val * 1.4))
        if val > 10000:
            break
        if val not in geo:
            geo.append(val)
    if len(geo) >= 20:
        initial_sets.append(geo[:20])
    
    # 5. Random from pool
    if len(candidate_pool) >= 20:
        initial_sets.append(sorted(random.sample(candidate_pool, 20)))
    
    # Evaluate all initial candidates
    best_A = None
    best_score = float('inf')
    
    for init_set in initial_sets:
        if time.time() > deadline - 10.0:
            break
        
        init_set = sorted(set(x for x in init_set if 1 <= x <= 10000))
        if len(init_set) < 20:
            for x in candidate_pool:
                if x not in init_set and x <= 10000:
                    init_set.append(x)
                    if len(init_set) == 20:
                        break
        init_set = sorted(init_set[:20])
        
        if len(init_set) != 20:
            continue
        
        score = _calc_score(init_set)
        if score < best_score:
            best_score = score
            best_A = list(init_set)
    
    # Phase 2: Local search from best initial
    if best_A is not None and time.time() < deadline - 15.0:
        remaining = min(30.0, deadline - time.time() - 10.0)
        best_A, best_score = _local_search(
            best_A, candidate_pool,
            max_iters=8000,
            timeout=remaining
        )
    
    # Phase 3: Polish pass
    if best_A is not None and time.time() < deadline - 5.0:
        remaining = min(10.0, deadline - time.time() - 2.0)
        best_A, best_score = _polish_pass(
            best_A,
            timeout=remaining
        )
    
    # Fallback
    if best_A is None or len(best_A) != 20:
        best_A = list(range(1, 21))
    
    return best_A
