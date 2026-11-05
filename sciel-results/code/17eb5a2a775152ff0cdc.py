import time
from math import isqrt

def _is_powerful(n):
    """Check if n is a powerful number (every prime factor appears with exponent >= 2)."""
    if n < 1:
        return False
    if n == 1:
        return True
    temp = n
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes:
        if p * p > temp:
            break
        if temp % (p * p) == 0:
            while temp % p == 0:
                temp //= p
        elif temp % p == 0:
            return False
    if temp > 1:
        r = isqrt(temp)
        if r * r != temp:
            return False
    return True

def _generate_powerful_numbers(limit):
    """Generate all powerful numbers up to limit."""
    powerful = set()
    powerful.add(1)
    
    # Powerful numbers are of the form a^2 * b^3 where b is square-free
    max_b = int(limit ** (1/3)) + 2
    
    # Pre-compute small primes for square-free check
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    for b in range(1, max_b + 1):
        t = b
        is_sf = True
        for p in small_primes:
            if p * p > t:
                break
            if t % (p * p) == 0:
                is_sf = False
                break
            while t % p == 0:
                t //= p
        if not is_sf:
            continue
        if t > 1:
            r = isqrt(t)
            if r * r == t:
                is_sf = False
        if not is_sf:
            continue
            
        b3 = b ** 3
        if b3 > limit:
            continue
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit
    
    # Generate powerful numbers up to a limit
    # We want to maximize the number of pairs a+b=N where a,b are powerful
    # The count of powerful numbers up to X is roughly ~ c * sqrt(X) for some constant
    # For X = 10^8, sqrt(X) = 10^4, so maybe ~20000-30000 numbers
    # For X = 10^7, sqrt(X) = 3162, so maybe ~5000-10000 numbers
    
    # The key insight: we want to find N with many representations as sum of two powerful numbers
    # This is like finding the "peak" in the convolution of the indicator function of powerful numbers
    
    # Strategy: Generate powerful numbers up to some limit L, then count all sums
    # If we have K powerful numbers, we do O(K^2) additions
    # For K=10000, that's 50M operations which is doable in ~5-10s in Python
    # For K=20000, that's 200M operations which is ~20-40s
    
    # Let's try L = 8*10^7 first, which should give ~15000-20000 powerful numbers
    # Then do the O(K^2) counting
    
    pow_limit = 8 * 10**7
    powerful = _generate_powerful_numbers(pow_limit)
    K = len(powerful)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        # Not enough time left for O(K^2), return a fallback
        # Find a small N with known representations
        return {'N': 5, 'pairs': [[1, 4]]}
    
    # Count sums using dictionary
    sum_counts = {}
    pow_list = powerful
    
    # Use a more efficient approach: for each pair (i,j) with i<=j, compute sum and count
    # This is O(K^2/2) operations
    
    # Optimization: use local variable references
    counts = sum_counts
    get = counts.get
    t0 = time.time()
    
    for i in range(K):
        a = pow_list[i]
        for j in range(i, K):
            b = pow_list[j]
            s = a + b
            counts[s] = get(s, 0) + 1
        # Check time every outer iteration
        if time.time() > deadline - 15:
            break
    
    if not counts:
        return {'N': 5, 'pairs': [[1, 4]]}
        
    best_N = max(counts, key=counts.get)
    best_count = counts[best_N]
    
    # Extract pairs
    pow_set = set(pow_list)
    pairs = []
    for a in pow_list:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in pow_set and a <= b:
            pairs.append([a, b])
            
    # Verify
    verified_pairs = []
    for p in pairs:
        if _is_powerful(p[0]) and _is_powerful(p[1]):
            verified_pairs.append(p)
            
    if not verified_pairs:
        verified_pairs = [[1, 4]]
        best_N = 5
        
    return {'N': best_N, 'pairs': verified_pairs}
