import time
import random
from itertools import combinations

def _calc_score(A):
    """Compute max(|A+A|, |A*A|) for set A."""
    A = list(A)
    sumset = set()
    prodset = set()
    for i, a in enumerate(A):
        for b in A[i:]:
            sumset.add(a + b)
            prodset.add(a * b)
    return max(len(sumset), len(prodset))

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
    for a in range(14):  # 2^13 = 8192
        for b in range(9):  # 3^8 = 6561
            for c in range(6):  # 5^5 = 3125
                val = (1 << a) * (3 ** b) * (5 ** c)
                if val <= n_max:
                    result.add(val)
    return sorted(result)

def _local_search(A, candidate_pool, max_iters=5000, timeout=50.0):
    """Local search to improve the set."""
    A = sorted(set(A))
    current_score = _calc_score(A)
    best_A = list(A)
    best_score = current_score
    start_time = time.time()
    
    for _ in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        # Try swapping one element
        if len(A) < 20:
            # Add a random element from pool
            candidates = [x for x in candidate_pool if x not in A]
            if candidates:
                new_elem = random.choice(candidates)
                new_A = sorted(A + [new_elem])
                new_score = _calc_score(new_A)
                if new_score < best_score:
                    best_A = new_A
                    best_score = new_score
                    A = new_A
        else:
            # Swap one element
            idx = random.randint(0, len(A) - 1)
            candidates = [x for x in candidate_pool if x not in A]
            if candidates:
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
    
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Generate candidate pools
    pool_2smooth = _generate_2smooth(10000)
    pool_3smooth = _generate_3smooth(10000)
    
    # Combine pools, ensure distinct
    candidate_pool = list(set(pool_2smooth + pool_3smooth))
    random.shuffle(candidate_pool)
    
    # Initial sets to try
    initial_sets = [
        list(range(1, 21)),
        sorted(pool_2smooth[:20]),
        sorted(pool_3smooth[:20]),
    ]
    
    best_A = None
    best_score = float('inf')
    
    for init_set in initial_sets:
        if time.time() > deadline:
            break
        
        # Ensure we have exactly 20 distinct positive integers <= 10000
        init_set = sorted(set(init_set))
        if len(init_set) < 20:
            # Fill from candidate pool
            for x in candidate_pool:
                if x not in init_set and x <= 10000:
                    init_set.append(x)
                    if len(init_set) == 20:
                        break
        init_set = sorted(init_set[:20])
        
        if len(init_set) != 20:
            continue
        
        # Run local search
        result_A, result_score = _local_search(
            init_set, candidate_pool, 
            max_iters=3000, 
            timeout=min(45.0, deadline - time.time() - 5.0)
        )
        
        if result_score < best_score:
            best_score = result_score
            best_A = result_A
        
        if time.time() > deadline - 10.0:
            break
    
    # If we didn't find a good set, use a default
    if best_A is None or len(best_A) != 20:
        best_A = list(range(1, 21))
    
    return best_A
# EVOLVE-BLOCK-END
