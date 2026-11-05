import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Precompute primes up to 10^6
    limit_p = 10**6
    is_prime = [True] * (limit_p + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    # 2. Sieve phi for small numbers (1 to 10^6)
    N_small = 10**6
    phi_small = list(range(N_small + 1))
    for i in range(2, N_small + 1):
        if phi_small[i] == i: # prime
            for j in range(i, N_small + 1, i):
                phi_small[j] -= phi_small[j] // i
                
    def phi_large(n):
        result = n
        y = n
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result

    best_k = 1
    best_s = 1
    
    # 3. Scan small range
    # We can optimize the distinct check
    for s in range(1, N_small):
        seen = set()
        k = 0
        # Max possible k is bounded by N_small - s
        while s + k <= N_small:
            v = phi_small[s + k]
            if v in seen:
                break
            seen.add(v)
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
        # Early termination check
        if time.time() > deadline * 0.5:
            break
            
    # 4. Scan larger range or random sample
    # If we have time, we can try to scan further or random sample
    # Random sampling is better for finding rare long runs in huge space
    
    if time.time() < deadline:
        # Try to scan a bit more linearly?
        # Or just random sample
        random.seed(42)
        
        # We can also try to extend the best found so far?
        # No, just sample new starts.
        
        # How many samples?
        # Each sample involves computing phi for ~k numbers.
        # If k is ~20, that's 20 phi computations.
        # phi_large is fast.
        
        while time.time() < deadline:
            # Pick a random start
            # Bias towards larger numbers?
            # Or uniform?
            s = random.randint(N_small + 1, 10**12 - 200)
            
            # Check run length
            seen = set()
            k = 0
            # Limit k to current best + 10 to save time?
            # If we find > best_k, we update.
            # If we find < best_k, we don't care about exact length if it's small.
            # But we need to know if it's > best_k.
            # So we check up to best_k + 1.
            # If we reach best_k + 1 without collision, we have a new best.
            # Then we can continue to find the full length.
            
            limit_k = best_k + 1
            for i in range(limit_k):
                n = s + i
                if n > 10**12:
                    break
                v = phi_large(n)
                if v in seen:
                    break
                seen.add(v)
                k += 1
            else:
                # We exceeded best_k + 1, so it's a new best
                # Continue to find full length
                while True:
                    n = s + k
                    if n > 10**12:
                        break
                    v = phi_large(n)
                    if v in seen:
                        break
                    seen.add(v)
                    k += 1
                    if time.time() > deadline:
                        break
                
                if k > best_k:
                    best_k = k
                    best_s = s

    return {'start': best_s, 'length': best_k}
