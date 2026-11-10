import time
import random
import math

def _calc_score(A):
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            sums.add(ai + A[j])
            prods.add(ai * A[j])
    return max(len(sums), len(prods))

def _generate_smooth(primes, max_val, count):
    vals = set([1])
    for p in primes:
        new_vals = set()
        for v in list(vals):
            x = v
            while x * p <= max_val:
                x *= p
                new_vals.add(x)
        vals.update(new_vals)
    vals = sorted(vals)
    return vals[:count]

def _generate_geometric_candidates(max_val=10000, n=20, k_max=3):
    candidates = []
    # Base geometric progressions
    for base in range(2, 10):
        # Find max exponent such that base^exp <= max_val
        exp_max = 0
        val = 1
        while val * base <= max_val:
            val *= base
            exp_max += 1
        if exp_max < n:
            continue
        
        # Generate a few exponent sets
        # 1. Consecutive: 0, 1, ..., n-1
        exponents = list(range(n))
        A = [base ** e for e in exponents]
        if max(A) <= max_val:
            candidates.append(A)
        
        # 2. Random exponents within range
        rng = random.Random(base)
        for _ in range(5):
            exponents = sorted(rng.sample(range(exp_max + 1), n))
            A = [base ** e for e in exponents]
            if max(A) <= max_val and len(set(A)) == n:
                candidates.append(A)
                
    # Multiplicative perturbation of small geometric sets
    # Start with small base GP and multiply by small coprime factors
    for base in [2, 3]:
        # Small GP: 1, base, base^2
        core = [1, base, base**2]
        # Try to extend this structure by multiplying elements by small integers
        # This is a heuristic to find "almost" geometric sets with collisions
        pass
        
    return candidates

def _local_search(A, timeout, candidate_pool):
    A = sorted(A)
    current_score = _calc_score(A)
    best_A = list(A)
    best_score = current_score
    start_time = time.time()
    n = len(A)
    max_val = 10000
    
    # Precompute sets for faster delta estimation if possible, 
    # but for simplicity and correctness, we use _calc_score for verification.
    # However, we can use a fast check for the candidate pool.
    
    while time.time() - start_time < timeout:
        A_set = set(A)
        best_new_score = current_score
        best_swap = None
        
        # We iterate through a subset of the candidate pool to save time
        # The pool should be sorted and we can limit the number of candidates per step
        # or just iterate all if pool is small enough.
        
        # To be safe, we iterate all candidates in the pool for each index.
        # If pool is large, this might be slow. 
        # Let's assume pool size is manageable (e.g., < 1000).
        
        for i in range(n):
            old_val = A[i]
            
            # Check time
            if time.time() - start_time > timeout:
                return sorted(best_A), best_score
                
            for v in candidate_pool:
                if v == old_val or v in A_set:
                    continue
                
                A[i] = v
                new_score = _calc_score(A)
                A[i] = old_val
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_swap = (i, v)
                    
        if best_swap is not None:
            i, v = best_swap
            A[i] = v
            current_score = best_new_score
            if current_score < best_score:
                best_score = current_score
                best_A = list(A)
        else:
            break
            
    return sorted(best_A), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin for 90s limit
    
    candidates = []
    
    # 1. Standard smooth numbers
    s5 = _generate_smooth([2, 3, 5, 7], 10000, 20)
    if len(s5) == 20:
        candidates.append(s5)
        
    s4 = _generate_smooth([2, 3, 5], 10000, 20)
    if len(s4) == 20:
        candidates.append(s4)
        
    # 2. Geometric Progressions (New Strategy)
    geos = _generate_geometric_candidates(10000, 20)
    for g in geos:
        if len(set(g)) == 20 and max(g) <= 10000:
            candidates.append(sorted(g))
            
    # 3. Powers of 2
    p2 = [2**i for i in range(20)]
    if max(p2) <= 10000:
        candidates.append(p2)
        
    # 4. Random
    random.seed(42)
    for _ in range(3):
        candidates.append(sorted(random.sample(range(1, 10001), 20)))
        
    # Build candidate pool for local search
    # Include small integers and smooth numbers to allow transitions
    pool_small = list(range(1, 101))
    pool_smooth = _generate_smooth([2, 3, 5, 7], 10000, 200)
    candidate_pool = sorted(set(pool_small + pool_smooth))
    
    best_A = None
    best_score = float('inf')
    
    # Sort candidates by initial score to prioritize good starts
    scored_candidates = []
    for cand in candidates:
        cand = sorted(set(x for x in cand if 1 <= x <= 10000))
        if len(cand) < 20:
            continue
        score = _calc_score(cand)
        scored_candidates.append((score, cand))
        
    scored_candidates.sort(key=lambda x: x[0])
    
    time_per_candidate = (deadline - time.time()) / len(scored_candidates)
    
    for score_init, cand in scored_candidates:
        if time.time() > deadline - 5.0:
            break
            
        if score_init < best_score:
            best_score = score_init
            best_A = list(cand)
            
        # Local search
        # Allocate time based on remaining time and number of remaining candidates
        # But ensure we don't starve later candidates if they are promising?
        # Since we sorted by initial score, earlier ones are more promising.
        # Let's give each candidate a fixed chunk of time, say 5-10 seconds.
        chunk_time = min(8.0, deadline - time.time() - 2.0)
        if chunk_time > 1.0:
            A_res, score_res = _local_search(cand, chunk_time, candidate_pool)
            if score_res < best_score:
                best_score = score_res
                best_A = A_res
                
    if best_A is None:
        best_A = list(range(1, 21))
        
    return best_A
