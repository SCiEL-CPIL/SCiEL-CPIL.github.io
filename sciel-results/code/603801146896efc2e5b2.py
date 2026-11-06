# EVOLVE-BLOCK-START
import time
from math import isqrt

def _is_powerful(n):
    """Check if n is powerful: every prime factor appears with exponent >= 2."""
    if n < 1:
        return False
    if n == 1:
        return True
    m = n
    d = 2
    while d * d <= m:
        if m % d == 0:
            cnt = 0
            while m % d == 0:
                m //= d
                cnt += 1
            if cnt < 2:
                return False
        d += 1
    if m > 1:
        return False  # remaining factor has exponent 1
    return True

def _gen_powerful(limit):
    """Generate all powerful numbers up to limit."""
    pw = set([1])
    # For each prime p, p^2, p^3, ... up to limit
    # Powerful numbers are of form prod(p_i^e_i) with e_i >= 2
    # Use DFS over primes
    primes = []
    is_prime = [True] * (isqrt(limit) + 2)
    for i in range(2, isqrt(limit) + 2):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, isqrt(limit) + 2, i):
                is_prime[j] = False
    
    def dfs(idx, current, max_p):
        if current > limit:
            return
        pw.add(current)
        for i in range(idx, len(primes)):
            p = primes[i]
            if p > max_p or current * p * p > limit:
                break
            # Try p^2, p^3, ...
            val = current * p * p
            while val <= limit:
                pw.add(val)
                # Recurse with next primes
                dfs(i + 1, val, p)
                val *= p
    
    dfs(0, 1, limit)
    return sorted(pw)

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin
    
    # Generate powerful numbers up to a reasonable limit
    # We want N up to 10^18, but let's start with smaller and see
    limit_pw = 10**7  # powerful numbers up to 10^7
    pw_list = _gen_powerful(limit_pw)
    pw_set = set(pw_list)
    
    # Now find N with max pairs where a+b=N, a<=b, a,b in pw_set
    # N ranges from 2 to 2*limit_pw
    # Use convolution-like approach: for each a in pw_list, for each b in pw_list with a<=b
    # count[a+b] += 1
    # But 10^4 * 10^4 = 10^8 which might be slow. Let's use numpy.
    
    import numpy as np
    
    # Better: for each a in pw_list, N-a must be in pw_set
    # Count pairs for each N
    # N ranges from 2 to 2*limit_pw
    max_N = 2 * limit_pw
    counts = np.zeros(max_N + 1, dtype=np.int32)
    
    for a in pw_list:
        if a > max_N // 2:
            break
        # b = N - a, need b >= a and b in pw_set
        # N = a + b, so b ranges from a to max_N - a
        # For each b in pw_list with b >= a and b <= max_N - a:
        #   counts[a + b] += 1
        # This is O(|pw_list|^2) in worst case
    
    # Alternative: use the fact that pw_list is sorted
    # For each a, find all b >= a in pw_list with a+b <= max_N
    for a in pw_list:
        if 2 * a > max_N:
            break
        # b ranges from a to max_N - a
        # Find indices in pw_list where b is in [a, max_N - a]
        lo = 0
        hi = len(pw_list)
        # Find first index where pw_list[idx] >= a
        while lo < hi:
            mid = (lo + hi) // 2
            if pw_list[mid] < a:
                lo = mid + 1
            else:
                hi = mid
        start_idx = lo
        # Find first index where pw_list[idx] > max_N - a
        lo2 = start_idx
        hi2 = len(pw_list)
        while lo2 < hi2:
            mid = (lo2 + hi2) // 2
            if pw_list[mid] <= max_N - a:
                lo2 = mid + 1
            else:
                hi2 = mid
        end_idx = lo2
        # For each b in pw_list[start_idx:end_idx], counts[a+b] += 1
        for b in pw_list[start_idx:end_idx]:
            counts[a + b] += 1
    
    # Find N with max count
    best_N = int(np.argmax(counts))
    best_count = int(counts[best_N])
    
    # Collect pairs for best_N
    pairs = []
    for a in pw_list:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in pw_set:
            pairs.append([a, b])
    
    # Verify with exact check
    verified_pairs = []
    for a, b in pairs:
        if a + b == best_N and a <= b and _is_powerful(a) and _is_powerful(b):
            verified_pairs.append([a, b])
    
    return {'N': best_N, 'pairs': verified_pairs}
# EVOLVE-BLOCK-END
