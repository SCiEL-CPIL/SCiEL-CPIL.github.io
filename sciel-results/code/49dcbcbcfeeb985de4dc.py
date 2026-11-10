import time
import random
import numpy as np

def _calc_score(A):
    """Compute max(|A+A|, |A*A|) for set A. Optimized for speed."""
    A = sorted(A)
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            sums.add(ai + A[j])
            prods.add(ai * A[j])
    return max(len(sums), len(prods))

def _calc_score_fast(A_list):
    """Compute max(|A+A|, |A*A|) for a Python list, optimized."""
    A = np.array(A_list, dtype=np.int64)
    n = len(A)
    i_idx, j_idx = np.triu_indices(n)
    sums = A[i_idx] + A[j_idx]
    prods = A[i_idx] * A[j_idx]
    sumset_size = len(np.unique(sums))
    prodset_size = len(np.unique(prods))
    return max(sumset_size, prodset_size)

def _generate_2smooth(n_max=10000):
    """Generate 2-smooth numbers (2^a * 3^b) up to n_max."""
    result = set()
    a = 0
    while (1 << a) <= n_max:
        b = 0
        val = 1 << a
        while val <= n_max:
            result.add(val)
            b += 1
            val = (1 << a) * (3 ** b)
        a += 1
    return sorted(result)

def _generate_3smooth(n_max=10000):
    """Generate 3-smooth numbers (2^a * 3^b * 5^c) up to n_max."""
    result = set()
    for a in range(14):
        for b in range(9):
            for c in range(6):
                val = (1 << a) * (3 ** b) * (5 ** c)
                if val <= n_max:
                    result.add(val)
    return sorted(result)

def _generate_4smooth(n_max=10000):
    """Generate 4-smooth numbers (2^a * 3^b * 5^c * 7^d) up to n_max."""
    result = set()
    for a in range(14):
        for b in range(9):
            for c in range(6):
                for d in range(4):
                    val = (1 << a) * (3 ** b) * (5 ** c) * (7 ** d)
                    if val <= n_max:
                        result.add(val)
    return sorted(result)

def _local_search(A, candidate_pool, max_iters=5000, timeout=40.0):
    """Local search: try all candidate swaps for each element position."""
    A = sorted(set(A))
    if len(A) != 20:
        return A, float('inf')
    
    best_A = list(A)
    best_score = _calc_score(A)
    start_time = time.time()
    
    A_set = set(A)
    pool_set = set(candidate_pool)
    
    it = 0
    while it < max_iters:
        if time.time() - start_time > timeout:
            break
        
        # Try swapping each element with every candidate not in A
        improved = False
        for idx in range(len(A)):
            old_val = A[idx]
            for new_val in candidate_pool:
                if new_val in A_set or new_val == old_val:
                    continue
                
                new_A = A[:idx] + [new_val] + A[idx+1:]
                new_A = sorted(new_A)
                
                new_score = _calc_score(new_A)
                if new_score < best_score:
                    best_A = new_A
                    best_score = new_score
                    A = new_A
                    A_set = set(A)
                    improved = True
                    break
            if time.time() - start_time > timeout:
                break
        
        it += 1
        if not improved:
            # No improvement this full pass; do a few random restarts
            for _ in range(3):
                if time.time() - start_time > timeout:
                    break
                idx = random.randint(0, len(A) - 1)
                candidates = [x for x in candidate_pool if x not in A_set and x != A[idx]]
                if not candidates:
                    break
                new_val = random.choice(candidates)
                new_A = A[:idx] + [new_val] + A[idx+1:]
                new_A = sorted(new_A)
                new_score = _calc_score(new_A)
                if new_score < best_score:
                    best_A = new_A
                    best_score = new_score
                    A = new_A
                    A_set = set(A)
                    improved = True
            if not improved:
                break
    
    return best_A, best_score

def _fine_polish(A, timeout=15.0):
    """Fine-grained polish: try small perturbations around each element."""
    A = sorted(set(A))
    if len(A) != 20:
        return A, float('inf')
    
    best_A = list(A)
    best_score = _calc_score(A)
    start_time = time.time()
    
    A_set = set(A)
    
    for it in range(20000):
        if time.time() - start_time > timeout:
            break
        
        # Pick a random element and try nearby values
        idx = random.randint(0, len(A) - 1)
        old_val = A[idx]
        
        # Try values in a window around old_val
        found_improvement = False
        for delta in range(1, 50):
            for new_val in (old_val - delta, old_val + delta):
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in A_set:
                    continue
                
                new_A = A[:idx] + [new_val] + A[idx+1:]
                new_A = sorted(new_A)
                
                new_score = _calc_score(new_A)
                if new_score < best_score:
                    best_A = new_A
                    best_score = new_score
                    A = new_A
                    A_set = set(A)
                    found_improvement = True
                    break
            if found_improvement:
                break
    
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Generate candidate pools
    pool_2smooth = _generate_2smooth(10000)
    pool_3smooth = _generate_3smooth(10000)
    pool_4smooth = _generate_4smooth(10000)
    
    # Combine pools
    candidate_pool = list(set(pool_2smooth + pool_3smooth + pool_4smooth))
    random.shuffle(candidate_pool)
    
    # Initial sets to try
    initial_sets = []
    
    # 1. Consecutive integers
    initial_sets.append(list(range(1, 21)))
    
    # 2. Smallest 2-smooth
    if len(pool_2smooth) >= 20:
        initial_sets.append(pool_2smooth[:20])
    
    # 3. Smallest 3-smooth
    if len(pool_3smooth) >= 20:
        initial_sets.append(pool_3smooth[:20])
    
    # 4. Smallest 4-smooth
    if len(pool_4smooth) >= 20:
        initial_sets.append(pool_4smooth[:20])
    
    # 5. Geometric progression with ratio ~1.3
    geo = []
    val = 1
    for i in range(20):
        val = int(round(val * 1.3))
        if val > 10000:
            break
        if val not in geo:
            geo.append(val)
    if len(geo) >= 20:
        initial_sets.append(geo[:20])
    
    # 6. Geometric progression with ratio ~1.5
    geo2 = []
    val = 1
    for i in range(20):
        val = int(round(val * 1.5))
        if val > 10000:
            break
        if val not in geo2:
            geo2.append(val)
    if len(geo2) >= 20:
        initial_sets.append(geo2[:20])
    
    # 7. Mix: small numbers + powers of 2
    mix = list(range(1, 11)) + [2**i for i in range(4, 14)]
    mix = sorted(set(x for x in mix if x <= 10000))
    if len(mix) >= 20:
        initial_sets.append(mix[:20])
    
    # 8. Random sample from smooth pool
    if len(candidate_pool) >= 20:
        initial_sets.append(sorted(random.sample(candidate_pool, 20)))
    
    # Evaluate all initial candidates
    best_A = None
    best_score = float('inf')
    
    for init_set in initial_sets:
        if time.time() > deadline - 5.0:
            break
        
        # Ensure validity
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
    
    # Phase 1: Local search from best initial
    if best_A is not None and time.time() < deadline - 20.0:
        remaining = min(40.0, deadline - time.time() - 15.0)
        best_A, best_score = _local_search(
            best_A, candidate_pool,
            max_iters=5000,
            timeout=remaining
        )
    
    # Phase 2: Fine polish
    if best_A is not None and time.time() < deadline - 10.0:
        remaining = min(15.0, deadline - time.time() - 5.0)
        best_A, best_score = _fine_polish(
            best_A,
            timeout=remaining
        )
    
    # Phase 3: Multi-restart with diverse seeds
    if best_A is not None and time.time() < deadline - 15.0:
        random.seed(42)
        for restart in range(8):
            if time.time() > deadline - 10.0:
                break
            
            # Generate diverse random sets
            if len(candidate_pool) >= 20:
                A_rand = sorted(random.sample(candidate_pool, 20))
            else:
                A_rand = sorted(random.sample(range(1, 10001), 20))
            
            score_rand = _calc_score(A_rand)
            
            # Also try a set biased toward smaller smooth numbers
            small_pool = [x for x in candidate_pool if x <= 500]
            if len(small_pool) >= 20:
                A_small = sorted(random.sample(small_pool, 20))
                score_small = _calc_score(A_small)
                if score_small < score_rand:
                    A_rand = A_small
                    score_rand = score_small
            
            if score_rand < best_score:
                best_score = score_rand
                best_A = list(A_rand)
                
                # Quick local search
                if time.time() < deadline - 10.0:
                    remaining = min(10.0, deadline - time.time() - 5.0)
                    best_A, best_score = _local_search(
                        best_A, candidate_pool,
                        max_iters=2000,
                        timeout=remaining
                    )
    
    # Phase 4: Final polish on best result
    if best_A is not None and time.time() < deadline - 5.0:
        remaining = min(5.0, deadline - time.time() - 2.0)
        if remaining > 1.0:
            best_A, best_score = _fine_polish(
                best_A,
                timeout=remaining
            )
    
    # Fallback
    if best_A is None or len(best_A) != 20:
        best_A = list(range(1, 21))
    
    return best_A
