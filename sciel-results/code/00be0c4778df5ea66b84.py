import time
import random
import math

def construct():
    # 1. Sieve of Eratosthenes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    def phi(x):
        if x == 1:
            return 1
        result = x
        y = x
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

    start_time = time.time()
    deadline = 100.0  # Leave 20s margin for 120s limit
    
    best_k = 1
    best_s = 1
    
    # Bias search towards larger numbers where collisions are rarer
    low_bound = 10**11
    high_bound = 10**12
    
    max_window = 200
    
    while time.time() - start_time < deadline:
        # Pick a random start
        s = random.randint(low_bound, high_bound - max_window + 1)
        
        seen = set()
        k_fwd = 0
        # Compute phi for s, s+1, ..., s+max_window-1
        # and check for distinctness
        for i in range(max_window):
            val = phi(s + i)
            if val in seen:
                break
            seen.add(val)
            k_fwd += 1
        
        # Extend backwards
        # We have phi(s) ... phi(s + k_fwd - 1) in seen
        # Check s-1, s-2, ...
        current_s = s
        current_k = k_fwd
        
        # We can extend backwards as long as we don't go below 1
        # and we stay within the "interesting" range or just 1..10^12
        # Since we are looking for max k, extending backwards is good.
        # But we must ensure s + k - 1 <= 10^12.
        # Our forward search already ensured s + k_fwd - 1 <= 10^12.
        # Extending backwards reduces s, so s_new + k_new - 1 = (s - delta) + (k + delta) - 1 = s + k - 1.
        # So the upper bound is preserved.
        
        j = s - 1
        while j >= 1:
            # Optional: stop if we go too far below low_bound?
            # No, valid range is 1..10^12.
            # But phi(j) for small j might be more prone to collisions?
            # It's fine, we just check.
            val = phi(j)
            if val in seen:
                break
            seen.add(val)
            current_k += 1
            current_s = j
            j -= 1
            
            # If we hit max length, break
            if current_k >= 200:
                break
                
        if current_k > best_k:
            best_k = current_k
            best_s = current_s
            if best_k == 200:
                break
                
    return {'start': best_s, 'length': best_k}
