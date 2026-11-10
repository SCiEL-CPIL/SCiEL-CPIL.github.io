import numpy as np
import time
import sys

MAX_VAL = 10_000_000
TIME_LIMIT = 85  # seconds, safety margin before 90s kill

def _sigma_sieve(n):
    """
    Vectorized NumPy sigma sieve: compute sigma(k) for all k in [1, n].
    sigma(k) = sum of divisors of k.
    Uses the fact that sigma is multiplicative and builds it via prime powers.
    """
    # Initialize sigma array with 1s (sigma(1) = 1)
    sigma = np.ones(n + 1, dtype=np.int64)
    
    # For each prime p, multiply sigma by (1 + p + p^2 + ... + p^e) for appropriate e
    # We use a sieve-like approach: for each prime p, iterate over multiples of p^1, p^2, ...
    
    # First, mark primes using a boolean sieve
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = False
    if n >= 1:
        is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    # For each prime, update sigma
    for p in range(2, n + 1):
        if not is_prime[p]:
            continue
        # For each power p^e, the factor for numbers divisible by p^e but not p^(e+1)
        # is (p^(e+1) - 1) / (p - 1)
        # We update sigma[m] for m divisible by p
        # Method: for each p, iterate m = p, 2p, 3p, ... and factor out p from m,
        # then sigma[m] = sigma[m // p^e] * (1 + p + ... + p^e)
        # But doing this naively is slow. Instead:
        # sigma[m] for m = p^e * r where gcd(r, p) = 1:
        # sigma[m] = sigma[r] * (1 + p + ... + p^e)
        
        # Alternative approach: use the fact that sigma is multiplicative.
        # Initialize sigma to 1, then for each prime p, for each power p^e:
        # For each r coprime to p, sigma[p^e * r] = sigma[r] * (1 + p + ... + p^e)
        # This is complex. Let's use a simpler sieve:
        # sigma[m] = sum_{d|m} d
        # We can compute this as: for each d from 1 to n, add d to sigma[m] for all m = d, 2d, 3d, ...
        # But that's O(n log n) which might be too slow for n=10^7 in NumPy.
        
        # Actually, the additive sieve: for d in 1..n: sigma[d::d] += d
        # This is O(n log n) ≈ 10^7 * ln(10^7) ≈ 1.6 * 10^8 operations
        # In NumPy, this should be fast enough.
        pass
    
    # Reset and use additive sieve
    sigma = np.zeros(n + 1, dtype=np.int64)
    for d in range(1, n + 1):
        sigma[d::d] += d
        if time.time() - start_time > TIME_LIMIT - 10:
            break
    
    return sigma

def construct():
    start_time = time.time()
    
    # Build sigma array using additive sieve
    # This takes O(n log n) but in NumPy it's vectorized
    
    # For efficiency, let's do the additive sieve in chunks or just let it run
    # n = 10^7, the loop is 10^7 iterations, each doing a vectorized addition
    # Total operations ≈ n * H(n) ≈ 10^7 * 16 ≈ 1.6e8, which should be a few seconds in NumPy
    
    # Actually, let's just do it
    sigma = np.zeros(MAX_VAL + 1, dtype=np.int64)
    for d in range(1, MAX_VAL + 1):
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        sigma[d::d] += d
    
    # Now search for pairs (a, b) with sigma[a] == sigma[b] and |a/b - 3/2| minimized
    # |a/b - 3/2| = |2a - 3b| / (2b)
    # We want to minimize |2a - 3b| / (2b)
    # For a given b, a should be close to 3b/2
    # Let's try k = |2a - 3b| for k = 1, 2, 3, ...
    # a = (3b + k)/2 or a = (3b - k)/2
    # a must be integer, so 3b + k must be even => b and k have same parity
    
    # Strategy: for small k, check if there exists b such that sigma[(3b+k)//2] == sigma[b]
    # We can build a dictionary: sigma_value -> list of indices
    # But that's memory intensive for 10^7 elements
    
    # Alternative: for each b from MAX_VAL down, check a = round(3b/2) candidates
    # and nearby values
    
    # Better: build a dict mapping sigma value to the max index and min index (or a set)
    # Then for each b, look up sigma[b] and check if any a with sigma[a] == sigma[b] is close to 3b/2
    
    # Let's build: sigma_to_indices = {sigma_val: sorted list of indices}
    # But this could use a lot of memory. Let's try a different approach.
    
    # For each k starting from 1, 2, 3, ...:
    #   For b from MAX_VAL down to 1 (with parity constraint):
    #     a1 = (3b + k) // 2 (if 3b + k is even)
    #     a2 = (3b - k) // 2 (if 3b - k is even and positive)
    #     Check if a1 or a2 is in [1, MAX_VAL] and sigma[a] == sigma[b]
    #   If found, return [a, b]
    #   Stop when time runs out
    
    # To speed up, precompute sigma as a numpy array and use direct indexing
    
    # First, find the max k we can try
    # For k=1: a ≈ 3b/2, error = 1/(2b). For b=10^7, error ≈ 5e-8
    # For k=2: error = 1/b = 1e-7 for b=10^7
    # So even k=1 with small b might be better than k=2 with large b?
    # No: error = k/(2b). So k=1, b=10^7 gives 5e-8. k=2, b=10^7 gives 1e-7.
    # So k=1 is always better for the same b.
    # We should try k=1 first with large b, then k=1 with smaller b, etc.
    
    # Actually, the optimal strategy is to minimize k/(2b), so we want small k and large b.
    # For k=1: best case is largest b where sigma[(3b+1)//2] == sigma[b] or sigma[(3b-1)//2] == sigma[b]
    # Note: 3b+1 is even iff b is odd. 3b-1 is even iff b is odd.
    # So for k=1, b must be odd.
    # a1 = (3b+1)//2, a2 = (3b-1)//2
    
    # For k=2: 3b+2 even iff b even. 3b-2 even iff b even.
    # So for k=2, b must be even.
    
    # Let's iterate k from 1 upward, and for each k, iterate b from MAX_VAL down.
    
    best_pair = (11, 6)  # fallback
    best_error = float('inf')
    
    for k in range(1, 1000):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        # Parity: 3b + k must be even => b + k must be even (since 3b has same parity as b)
        # So b and k must have same parity.
        if k % 2 == 1:
            start_b = MAX_VAL if MAX_VAL % 2 == 1 else MAX_VAL - 1
            step = -2
        else:
            start_b = MAX_VAL
            step = -1
        
        for b in range(start_b, 0, step):
            if time.time() - start_time > TIME_LIMIT - 2:
                return [int(best_pair[0]), int(best_pair[1])]
            
            # Case 1: a = (3b + k) // 2
            num1 = 3 * b + k
            if num1 % 2 == 0:
                a1 = num1 // 2
                if 1 <= a1 <= MAX_VAL:
                    if sigma[a1] == sigma[b]:
                        err = k / (2.0 * b)
                        if err < best_error:
                            best_error = err
                            best_pair = (a1, b)
                        return [int(a1), int(b)]
            
            # Case 2: a = (3b - k) // 2
            num2 = 3 * b - k
            if num2 > 0 and num2 % 2 == 0:
                a2 = num2 // 2
                if 1 <= a2 <= MAX_VAL:
                    if sigma[a2] == sigma[b]:
                        err = k / (2.0 * b)
                        if err < best_error:
                            best_error = err
                            best_pair = (a2, b)
                        return [int(a2), int(b)]
    
    return [int(best_pair[0]), int(best_pair[1])]
