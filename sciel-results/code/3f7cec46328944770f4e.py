import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin from 90s limit
    
    n = 12
    max_val = 10000
    
    # Precompute prime factors for all possible sums (2..20000)
    # Use a sieve-like approach for speed
    pf_cache = [set() for _ in range(2 * max_val + 1)]
    for p in range(2, 2 * max_val + 1):
        is_prime = True
        if p >= 4 and p % 2 == 0:
            is_prime = False
        elif p >= 9 and p % 3 == 0:
            is_prime = False
        elif p >= 25 and p % 5 == 0:
            is_prime = False
        elif p >= 49 and p % 7 == 0:
            is_prime = False
        elif p >= 121 and p % 11 == 0:
            is_prime = False
        elif p >= 169 and p % 13 == 0:
            is_prime = False
        elif p >= 289 and p % 17 == 0:
            is_prime = False
        elif p >= 361 and p % 19 == 0:
            is_prime = False
        elif p >= 529 and p % 23 == 0:
            is_prime = False
        elif p >= 841 and p % 29 == 0:
            is_prime = False
        elif p >= 1089 and p % 31 == 0:
            is_prime = False
        elif p >= 1369 and p % 37 == 0:
            is_prime = False
        elif p >= 1681 and p % 41 == 0:
            is_prime = False
        elif p >= 2025 and p % 43 == 0:
            is_prime = False
        elif p >= 2401 and p % 47 == 0:
            is_prime = False
        elif p >= 2809 and p % 53 == 0:
            is_prime = False
        elif p >= 3249 and p % 57 == 0: # 57 not prime, but check logic
            pass # This manual check is getting messy and error-prone for large p.
        
    # Better approach for pf_cache: simple trial division for each sum, but optimized
    # Or even better: build it by iterating primes and marking multiples.
    # Since sums are up to 20000, primes up to 20000 are needed.
    
    def generate_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit+1, i):
                    sieve[j] = False
        return [i for i in range(2, limit+1) if sieve[i]]
    
    primes = generate_primes(2 * max_val)
    
    # Re-initialize pf_cache correctly
    pf_cache = [set() for _ in range(2 * max_val + 1)]
    for p in primes:
        for multiple in range(p, 2 * max_val + 1, p):
            pf_cache[multiple].add(p)
    
    def count_distinct_prime_factors(arr):
        all_factors = set()
        for i in range(n):
            ai = arr[i]
            for j in range(i+1, n):
                s = ai + arr[j]
                all_factors.update(pf_cache[s])
        return len(all_factors)
    
    # Strategy 1: Structured Initialization (Modulus 210)
    # Sums of numbers congruent to specific residues mod 210 can be controlled.
    # If we pick numbers all congruent to 105 mod 210, sums are 210 mod 420.
    # 210 = 2*3*5*7. So sums are divisible by 2,3,5,7.
    # Let's try a mix to minimize other factors.
    
    best_score = float('inf')
    best_result = None
    
    # Candidate 1: Arithmetic progression with step 210
    # x_i = 105 + 210*i. Sum = 210 + 210*(i+j) = 210*(1+i+j).
    # Primes dividing 210 are 2,3,5,7. Other primes come from (1+i+j).
    # i, j in 0..11. 1+i+j ranges from 2 to 23.
    # Primes in 2..23: 2,3,5,7,11,13,17,19,23.
    # Union: {2,3,5,7,11,13,17,19,23} -> 9 primes.
    
    cand1 = [105 + 210*i for i in range(12)]
    if all(x <= 10000 for x in cand1):
        sc = count_distinct_prime_factors(cand1)
        if sc < best_score:
            best_score = sc
            best_result = cand1[:]
    
    # Candidate 2: Multiples of 210
    # x_i = 210*i. Sum = 210*(i+j).
    # i, j in 1..12. i+j ranges 3..23.
    # Primes in 3..23: 3,5,7,11,13,17,19,23.
    # Plus 2 from 210.
    # Union: {2,3,5,7,11,13,17,19,23} -> 9 primes.
    
    cand2 = [210*i for i in range(1, 13)]
    if all(x <= 10000 for x in cand2):
        sc = count_distinct_prime_factors(cand2)
        if sc < best_score:
            best_score = sc
            best_result = cand2[:]
    
    # Candidate 3: Powers of 2
    cand3 = [2**i for i in range(1, 13)]
    if all(x <= 10000 for x in cand3):
        sc = count_distinct_prime_factors(cand3)
        if sc < best_score:
            best_score = sc
            best_result = cand3[:]
    
    # Candidate 4: Smooth numbers (2,3,5,7-smooth)
    # Generate 7-smooth numbers up to 10000
    smooths = set()
    for a in range(0, 14): # 2^13 = 8192
        for b in range(0, 9): # 3^8 = 6561
            for c in range(0, 6): # 5^5 = 3125
                for d in range(0, 5): # 7^4 = 2401
                    v = (2**a) * (3**b) * (5**c) * (7**d)
                    if v <= 10000:
                        smooths.add(v)
    smooths_list = sorted(smooths)
    if len(smooths_list) >= 12:
        # Try the first 12
        cand4 = smooths_list[:12]
        sc = count_distinct_prime_factors(cand4)
        if sc < best_score:
            best_score = sc
            best_result = cand4[:]
            
        # Try a spread out subset of smooths
        # Pick evenly spaced indices
        step = len(smooths_list) // 12
        cand5 = [smooths_list[i*step] for i in range(12)]
        if len(set(cand5)) == 12:
            sc = count_distinct_prime_factors(cand5)
            if sc < best_score:
                best_score = sc
                best_result = cand5[:]
    
    # Local Search / Hill Climbing
    # Use the smooths list as the pool for replacements, as they are likely to have fewer prime factors in sums.
    # Also include small numbers.
    pool = smooths_list
    # Add some small numbers to pool for diversity
    for i in range(1, 50):
        if i not in pool:
            pool.append(i)
    pool = sorted(set(pool))
    
    current = best_result[:]
    
    # Optimization: Precompute pair sums and their factor sets?
    # No, just use pf_cache.
    
    def try_improve(arr, max_iters=10000):
        nonlocal best_score, best_result
        cur_score = count_distinct_prime_factors(arr)
        if cur_score < best_score:
            best_score = cur_score
            best_result = arr[:]
            
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 2.0:
            improved = False
            iters += 1
            # Try replacing each element
            for i in range(n):
                if time.time() > deadline - 2.0:
                    return
                old_val = arr[i]
                existing = set(arr)
                # Try candidates from pool
                # Sort pool by "smoothness" or just random?
                # Let's try a random sample from pool to avoid bias
                random.shuffle(pool)
                for new_val in pool:
                    if new_val == old_val:
                        continue
                    if new_val in existing:
                        continue
                    if new_val > max_val:
                        break
                    arr[i] = new_val
                    new_score = count_distinct_prime_factors(arr)
                    if new_score < cur_score:
                        cur_score = new_score
                        improved = True
                        if cur_score < best_score:
                            best_score = cur_score
                            best_result = arr[:]
                        break
                    else:
                        arr[i] = old_val
                if improved:
                    break
        return cur_score
    
    # Run local search from best initial candidate
    if best_result is not None:
        try_improve(best_result[:], 5000)
        
    # Random Restarts
    random.seed(42)
    for _ in range(20):
        if time.time() > deadline - 10.0:
            break
        # Random subset from pool
        if len(pool) >= 12:
            cand = random.sample(pool, 12)
            sc = count_distinct_prime_factors(cand)
            if sc < best_score:
                best_score = sc
                best_result = cand[:]
                try_improve(cand[:], 2000)
        
        # Random subset from 1..10000
        cand = random.sample(range(1, 10001), 12)
        sc = count_distinct_prime_factors(cand)
        if sc < best_score:
            best_score = sc
            best_result = cand[:]
            try_improve(cand[:], 2000)
    
    return best_result
