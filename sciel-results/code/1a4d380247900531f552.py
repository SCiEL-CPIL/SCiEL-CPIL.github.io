import time
import random

def construct():
    start_time = time.time()
    # Allow 85 seconds out of 90
    deadline = start_time + 85.0
    
    # 1. Precompute prime factors for sums up to 20000
    max_sum = 20000
    # Use a list of sets for prime factors
    prime_factors = [set() for _ in range(max_sum + 1)]
    # Sieve to find smallest prime factor, then factorize?
    # Or just iterate primes and add to multiples.
    # The latter is O(N log log N) roughly.
    for p in range(2, max_sum + 1):
        if not prime_factors[p]: # p is prime
            for multiple in range(p, max_sum + 1, p):
                prime_factors[multiple].add(p)
                
    # 2. Generate pool of 7-smooth numbers <= 10000
    pool = set()
    # 2^a 3^b 5^c 7^d
    for a in range(14):
        v1 = 2**a
        if v1 > 10000: break
        for b in range(10):
            v2 = v1 * (3**b)
            if v2 > 10000: break
            for c in range(8):
                v3 = v2 * (5**c)
                if v3 > 10000: break
                for d in range(6):
                    v4 = v3 * (7**d)
                    if v4 > 10000: break
                    if v4 >= 1:
                        pool.add(v4)
    
    pool = list(pool)
    # Sort pool to have small numbers first
    pool.sort()
    
    # 3. Initial Set: 12 smallest 7-smooth numbers
    current = pool[:12]
    current_set = set(current)
    
    def get_cost(arr):
        s = set()
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                val = arr[i] + arr[j]
                s.update(prime_factors[val])
        return len(s)
        
    current_cost = get_cost(current)
    best_cost = current_cost
    best_arr = current[:]
    
    # 4. Local Search
    # We will perform swaps.
    # To improve diversity, we can occasionally pick a random element from the entire pool, 
    # not just nearby ones.
    
    # Optimization: 
    # Instead of checking all pool elements for a swap, we can check a random subset.
    # Or iterate through pool.
    # Pool size ~ 300.
    # 12 elements.
    # 12 * 300 = 3600 possible swaps per "pass".
    # Each swap evaluation is ~100 ops.
    # 360,000 ops per pass.
    # Python can do ~10M-50M simple ops/sec.
    # So one pass takes ~0.01 - 0.05 sec.
    # We can do many passes.
    
    no_improve_count = 0
    max_no_improve = 50 # After 50 passes without improvement, do a random restart or kick
    
    while time.time() < deadline:
        improved = False
        # Iterate through all possible swaps
        # To avoid bias, we can shuffle the order of indices
        indices = list(range(12))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
                
            # Try swapping current[i] with candidates in pool
            # We can limit candidates to a random subset to speed up?
            # Or just iterate all.
            # Let's iterate all pool elements not in current_set.
            
            # Optimization: Pre-compute list of available candidates?
            # current_set changes.
            # Let's just check `if c in current_set: continue`
            
            for c in pool:
                if c in current_set:
                    continue
                
                old_val = current[i]
                # Swap
                current[i] = c
                current_set.remove(old_val)
                current_set.add(c)
                
                new_cost = get_cost(current)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_arr = current[:]
                        # We can break early if we reach a very low cost?
                        # Theoretical min is 2 (primes 2,3).
                        # If we hit 2, we can return?
                        # But maybe 1 is possible? No, sum of two distinct positives >= 1+2=3.
                        # So min primes is at least 1 (if all sums are powers of 2? 
                        # 1+3=4, 2+2 not allowed. 
                        # If we pick 1, 3, 5, 7... sums are even.
                        # 1+3=4 (2), 1+5=6 (2,3). So 2 and 3.
                        # Can we have only 2?
                        # Sums must be powers of 2.
                        # a_i + a_j = 2^k.
                        # This is a strong constraint.
                        # Let's assume min is 2 or 3.
                        if best_cost <= 2:
                            # Likely optimal
                            return best_arr
                else:
                    # Revert
                    current[i] = old_val
                    current_set.remove(c)
                    current_set.add(old_val)
                    
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Random Restart: Pick 12 random from pool
                # Or Perturb: Change 2 random elements
                # Let's perturb
                for _ in range(2):
                    idx = random.randint(0, 11)
                    old = current[idx]
                    # Pick random from pool
                    while True:
                        new_c = random.choice(pool)
                        if new_c not in current_set:
                            break
                    current[idx] = new_c
                    current_set.remove(old)
                    current_set.add(new_c)
                    # Update cost (might be worse, that's ok)
                    current_cost = get_cost(current)
                no_improve_count = 0
        else:
            no_improve_count = 0
            
    return best_arr
