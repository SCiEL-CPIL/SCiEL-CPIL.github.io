import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s buffer for safety
    
    # Precompute primes up to 20000
    limit = 20000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    # Fast factorization using sieve
    # We just need the set of prime factors
    # Precompute smallest prime factor (SPF) for speed
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_prime_factors(n):
        factors = set()
        while n > 1:
            factors.add(spf[n])
            n //= spf[n]
        return factors

    def evaluate(a):
        primes_union = set()
        n = len(a)
        for i in range(n):
            for j in range(i + 1, n):
                s = a[i] + a[j]
                primes_union.update(get_prime_factors(s))
        return len(primes_union)

    # Strategy: Use modulus M to constrain sums
    # Try M = 210 (2*3*5*7)
    # a_i = 105 + 210 * k_i
    # Max k: (10000 - 105) // 210 = 47
    # We search for k_i in {0..47}
    
    best_score = float('inf')
    best_a = []
    
    # Initial candidate
    k_vals = list(range(12))
    a_curr = [105 + 210 * k for k in k_vals]
    curr_score = evaluate(a_curr)
    best_score = curr_score
    best_a = a_curr[:]
    
    # Local Search
    # We represent the state as a set of k's
    # k_set = set(k_vals)
    
    def get_a_from_k(k_set):
        return [105 + 210 * k for k in sorted(k_set)]
    
    k_set = set(k_vals)
    a_curr = get_a_from_k(k_set)
    curr_score = evaluate(a_curr)
    
    # Hill Climbing
    # We can try to swap one k with another
    # Or just iterate through all possible k replacements
    
    # To speed up, we can cache scores of k sets? No, too many.
    # Just evaluate on the fly.
    
    # We will try to improve by replacing one k_i with a new k
    # There are 48 possible k values.
    
    # We can also try random restarts with different initial k sets
    # But the structure M=210 is strong.
    
    # Let's run a simple hill climbing
    improved = True
    while improved and time.time() < deadline:
        improved = False
        k_list = sorted(k_set)
        for idx, k_old in enumerate(k_list):
            # Try replacing k_old with k_new
            for k_new in range(48):
                if k_new == k_old or k_new in k_set:
                    continue
                
                # Swap
                k_set.remove(k_old)
                k_set.add(k_new)
                
                a_temp = get_a_from_k(k_set)
                score = evaluate(a_temp)
                
                if score < curr_score:
                    curr_score = score
                    if score < best_score:
                        best_score = score
                        best_a = a_temp[:]
                    improved = True
                else:
                    # Revert
                    k_set.remove(k_new)
                    k_set.add(k_old)
                    
                if time.time() > deadline - 5:
                    # Restore state before exiting
                    if k_new in k_set and k_old not in k_set:
                        k_set.remove(k_new)
                        k_set.add(k_old)
                    return best_a
    
    # If we have time, try M=30
    # a_i = 15 + 30 * k_i
    # Max k: (10000 - 15) // 30 = 332
    # This has more search space, but base primes are only 3.
    # If we can get co-factors to add 0 or 1 primes, total 3 or 4.
    # Let's try a quick search here too.
    
    if time.time() < deadline - 10:
        # Initialize with small k
        k_vals_30 = list(range(12))
        k_set_30 = set(k_vals_30)
        a_curr_30 = [15 + 30 * k for k in sorted(k_set_30)]
        curr_score_30 = evaluate(a_curr_30)
        
        if curr_score_30 < best_score:
            best_score = curr_score_30
            best_a = a_curr_30[:]
            
        improved_30 = True
        while improved_30 and time.time() < deadline - 5:
            improved_30 = False
            k_list_30 = sorted(k_set_30)
            for idx, k_old in enumerate(k_list_30):
                # Try a few random new k's to avoid full 332 loop if slow
                # But 332 is small. Let's try all.
                for k_new in range(333):
                    if k_new == k_old or k_new in k_set_30:
                        continue
                    
                    k_set_30.remove(k_old)
                    k_set_30.add(k_new)
                    
                    a_temp = [15 + 30 * k for k in sorted(k_set_30)]
                    score = evaluate(a_temp)
                    
                    if score < curr_score_30:
                        curr_score_30 = score
                        if score < best_score:
                            best_score = score
                            best_a = a_temp[:]
                        improved_30 = True
                    else:
                        k_set_30.remove(k_new)
                        k_set_30.add(k_old)
                    
                    if time.time() > deadline - 5:
                        if k_new in k_set_30 and k_old not in k_set_30:
                            k_set_30.remove(k_new)
                            k_set_30.add(k_old)
                        return best_a

    return best_a

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()

def _construct_impl():
    import time
    start_time = time.time()
    deadline = start_time + 80.0
    
    limit = 20000
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_prime_factors(n):
        factors = set()
        while n > 1:
            factors.add(spf[n])
            n //= spf[n]
        return factors

    def evaluate(a):
        primes_union = set()
        n = len(a)
        for i in range(n):
            ai = a[i]
            for j in range(i + 1, n):
                s = ai + a[j]
                primes_union.update(get_prime_factors(s))
        return len(primes_union)

    best_score = float('inf')
    best_a = []
    
    # M=210 Strategy
    # a_i = 105 + 210 * k_i
    # k in 0..47
    
    # Initial: 0..11
    k_set = set(range(12))
    a_curr = [105 + 210 * k for k in sorted(k_set)]
    curr_score = evaluate(a_curr)
    best_score = curr_score
    best_a = a_curr[:]
    
    def try_search(k_set, base, mod, max_k, current_score):
        nonlocal best_score, best_a
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            k_list = sorted(k_set)
            for idx, k_old in enumerate(k_list):
                for k_new in range(max_k + 1):
                    if k_new == k_old or k_new in k_set:
                        continue
                    
                    k_set.remove(k_old)
                    k_set.add(k_new)
                    
                    a_temp = [base + mod * k for k in sorted(k_set)]
                    score = evaluate(a_temp)
                    
                    if score < current_score:
                        current_score = score
                        if score < best_score:
                            best_score = score
                            best_a = a_temp[:]
                        improved = True
                    else:
                        k_set.remove(k_new)
                        k_set.add(k_old)
                        
                    if time.time() > deadline - 5:
                        if k_new in k_set and k_old not in k_set:
                            k_set.remove(k_new)
                            k_set.add(k_old)
                        return current_score
        
        return current_score

    # Run search for M=210
    curr_score = try_search(k_set, 105, 210, 47, curr_score)
    
    # Run search for M=30 if time permits
    if time.time() < deadline - 15:
        k_set_30 = set(range(12))
        a_curr_30 = [15 + 30 * k for k in sorted(k_set_30)]
        curr_score_30 = evaluate(a_curr_30)
        if curr_score_30 < best_score:
            best_score = curr_score_30
            best_a = a_curr_30[:]
        try_search(k_set_30, 15, 30, 332, curr_score_30)
        
    return best_a
# EVOLVE-BLOCK-END
