import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    LIMIT = 115.0  # seconds, leaving 5s margin for 120s limit
    MAX_S = 10**12
    MAX_K = 200

    # 1. Generate primes up to 10^6 for individual factorization
    limit_p = 10**6
    is_prime = [True] * (limit_p + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]

    # 2. Efficient phi calculation for a single number
    def phi(x):
        if x == 1:
            return 1
        result, m = x, x
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                result -= result // p
                while m % p == 0:
                    m //= p
        if m > 1:
            result -= result // m
        return result

    # 3. Segmented phi sieve for a window [s, s+K-1]
    #    Returns list of phi values for each number in the window
    def segmented_phi(s, K):
        # phi[i] will hold the totient of (s+i)
        phi_arr = np.array([s + i for i in range(K)], dtype=np.int64)
        
        # For each prime p, mark multiples in the window
        # For a number n divisible by p, phi(n) = n * (1 - 1/p)
        # We accumulate the product of (1 - 1/p) for each prime factor
        
        # Start with the identity: result[i] = n_i
        result = phi_arr.copy()
        remaining = phi_arr.copy()  # We'll divide out prime factors
        
        max_val = s + K - 1
        for p in primes:
            if p * p > max_val:
                break
            # Find the first multiple of p in [s, s+K-1]
            first = ((s + p - 1) // p) * p
            # Only process if first multiple is within range
            if first > max_val:
                continue
            for idx in range(first - s, K, p):
                if remaining[idx] % p == 0:
                    # Apply the (1 - 1/p) factor
                    result[idx] -= result[idx] // p
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
        
        # After processing all primes up to sqrt(max_val), any remaining > 1 is a prime factor
        for i in range(K):
            if remaining[i] > 1:
                result[i] -= result[i] // remaining[i]
        
        return result

    # 4. Helper: find longest run of distinct values in a list
    def longest_distinct_run(vals):
        n = len(vals)
        if n == 0:
            return 0, 0
        max_len = 1
        max_start = 0
        cur_start = 0
        seen = {}
        for i in range(n):
            v = vals[i]
            if v in seen and seen[v] >= cur_start:
                cur_start = seen[v] + 1
            seen[v] = i
            cur_len = i - cur_start + 1
            if cur_len > max_len:
                max_len = cur_len
                max_start = cur_start
        return max_len, max_start

    # 5. Check a window starting at s using segmented sieve
    #    Return (all_distinct, longest_k, longest_offset)
    def check_window(s):
        phis = segmented_phi(s, MAX_K)
        phis_list = list(phis)
        if len(set(phis_list)) == MAX_K:
            return True, MAX_K, 0
        k, offset = longest_distinct_run(phis_list)
        return False, k, offset

    # 6. Fast primality check using precomputed sieve (for numbers up to ~10^6)
    #    For larger numbers, use Miller-Rabin
    def is_prime_fast(n):
        """Fast primality test. Uses sieve for n <= limit_p, Miller-Rabin for larger."""
        if n < 2:
            return False
        if n <= limit_p:
            return is_prime[n]
        # Miller-Rabin for larger numbers
        if n % 2 == 0:
            return n == 2
        # Write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        # Test with witnesses
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    # 7. Probabilistic collision model: score a window by estimated collision risk
    #    Key insight: phi(n) = phi(n-1) often when:
    #    - n is even, n-1 is prime (phi(n-1) = n-2, phi(n) could be n/2 or other)
    #    - Both n and n-1 have similar small prime factor structures
    #    We estimate risk by checking for "dangerous" patterns
    
    def collision_risk(s, K):
        """Estimate collision risk for window [s, s+K-1]. Lower is safer."""
        risk = 0
        # Check for n where n-1 is prime (high collision risk with n even)
        # For each i in window, check if (s+i-1) is prime and (s+i) is even
        for i in range(1, K):
            n = s + i
            n_prev = n - 1
            if n_prev > 1 and n % 2 == 0:
                if is_prime_fast(n_prev):
                    risk += 10  # High risk: phi(n-1) = n-2, phi(n) likely to collide
        # Also check for n where n is prime and n+1 is even (symmetric case)
        for i in range(K - 1):
            n = s + i
            n_next = n + 1
            if n > 1 and n_next % 2 == 0:
                if is_prime_fast(n):
                    risk += 10
        return risk

    # 8. Search strategy:
    #    Phase 1: Deterministic candidates near 10^12
    #    Phase 2: Targeted search in low-risk intervals using collision model
    #    Phase 3: Random sampling as fallback
    
    best_k = 1
    best_s = 1

    # Phase 1: Deterministic candidates near 10^12
    deterministic_offsets = [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000]
    for offset in deterministic_offsets:
        if time.time() - start_time > LIMIT:
            break
        s = MAX_S - offset
        if s < MAX_K:
            continue
        all_distinct, k, off = check_window(s)
        if all_distinct:
            return {'start': s, 'length': MAX_K}
        if k > best_k:
            best_k = k
            best_s = s + off

    # Phase 2: Targeted search using collision risk model
    # Sample candidate windows, score them, and verify the safest ones first
    rng = random.Random(42)
    low = 10**11
    high = MAX_S - MAX_K
    
    # Generate a batch of candidates and sort by risk
    # Use larger batches and more diverse sampling
    candidates = []
    batch_size = 100
    for _ in range(batch_size):
        if time.time() - start_time > LIMIT:
            break
        # Sample from different ranges to cover more of the search space
        region = rng.random()
        if region < 0.3:
            s = rng.randint(10**10, 10**11)
        elif region < 0.6:
            s = rng.randint(10**11, 5 * 10**11)
        elif region < 0.9:
            s = rng.randint(5 * 10**11, 9 * 10**11)
        else:
            s = rng.randint(9 * 10**11, high)
        if s < MAX_K:
            continue
        risk = collision_risk(s, MAX_K)
        candidates.append((risk, s))
    
    # Sort by risk (lowest first) and verify
    candidates.sort()
    for risk, s in candidates:
        if time.time() - start_time > LIMIT:
            break
        all_distinct, k, off = check_window(s)
        if all_distinct:
            return {'start': s, 'length': MAX_K}
        if k > best_k:
            best_k = k
            best_s = s + off

    # Phase 3: Random sampling fallback with adaptive strategy
    # Focus on regions where we found good results
    search_regions = [(low, high)]
    while time.time() - start_time < LIMIT:
        # Pick a random region and sample from it
        region_low, region_high = rng.choice(search_regions)
        s = rng.randint(region_low, region_high)
        all_distinct, k, off = check_window(s)
        if all_distinct:
            return {'start': s, 'length': MAX_K}
        if k > best_k:
            best_k = k
            best_s = s + off
            # Add neighborhood of good result to search regions
            new_low = max(low, s - 10000)
            new_high = min(high, s + 10000)
            if new_low < new_high:
                search_regions.append((new_low, new_high))
                # Keep list manageable
                if len(search_regions) > 10:
                    search_regions = search_regions[-10:]

    # 9. Return best found
    return {'start': best_s, 'length': best_k}
