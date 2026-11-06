import time
import numpy as np
from math import isqrt, gcd
from collections import defaultdict

def _is_powerful(n):
    """Check if n is a powerful number (every prime factor appears with exponent >= 2).
    1 counts as powerful."""
    if n == 1:
        return True
    if n < 1:
        return False
    x = n
    d = 2
    while d * d <= x:
        if x % d == 0:
            cnt = 0
            while x % d == 0:
                x //= d
                cnt += 1
            if cnt < 2:
                return False
        d += 1
    if x > 1:
        return False
    return True

def _generate_powerful_numbers(limit):
    """Generate all powerful numbers up to limit using the form a^2 * b^3,
    then verify each is actually powerful."""
    pw_set = set()
    max_b = int(limit ** (1.0/3.0)) + 2
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= limit:
                pw_set.add(n)
    # Verify each candidate is truly powerful
    pw_list = sorted(n for n in pw_set if _is_powerful(n))
    return pw_list

def _factorize(n):
    """Return list of (prime, exponent) pairs for n."""
    if n == 1:
        return []
    factors = []
    x = n
    d = 2
    while d * d <= x:
        if x % d == 0:
            cnt = 0
            while x % d == 0:
                x //= d
                cnt += 1
            factors.append((d, cnt))
        d += 1
    if x > 1:
        factors.append((x, 1))
    return factors

def _is_powerful_fast(n, small_primes):
    """Faster powerful check using precomputed small primes."""
    if n == 1:
        return True
    if n < 1:
        return False
    x = n
    for p in small_primes:
        if p * p > x:
            break
        if x % p == 0:
            cnt = 0
            while x % p == 0:
                x //= p
                cnt += 1
            if cnt < 2:
                return False
    if x > 1:
        # x is a prime factor with exponent 1
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin for 150s limit
    
    # Structural change: Use a two-phase approach
    # Phase 1: Generate powerful numbers up to a moderate limit
    # Phase 2: Use collision counting with smart N selection
    
    # Generate small primes for fast powerful checking
    small_primes = []
    for d in range(2, 10000):
        is_prime = True
        for p in small_primes:
            if p * p > d:
                break
            if d % p == 0:
                is_prime = False
                break
        if is_prime:
            small_primes.append(d)
    
    # Phase 1: Generate powerful numbers up to L
    L = 2 * 10**7
    pw_list = _generate_powerful_numbers(L)
    pw_set = set(pw_list)
    
    elapsed = time.time() - start_time
    print(f"Phase 1: {len(pw_list)} powerful numbers up to {L}, took {elapsed:.1f}s")
    
    if elapsed > 40:
        # Fallback to smaller search
        L = 5 * 10**6
        pw_list = _generate_powerful_numbers(L)
        pw_set = set(pw_list)
    
    # Phase 2: Find N with maximum representations
    # Use FFT convolution for efficient counting
    max_N = 2 * L
    arr = np.zeros(max_N + 1, dtype=np.int32)
    for p in pw_list:
        if p <= max_N:
            arr[p] = 1
    
    from scipy.signal import fftconvolve
    counts = fftconvolve(arr, arr, mode='full')
    
    # Find top candidate N values
    best_N = 2
    best_count = 0
    second_best_N = 2
    second_best_count = 0
    
    for N in range(2, min(max_N + 1, len(counts))):
        ordered = int(counts[N])
        if N % 2 == 0 and (N // 2) in pw_set:
            unordered = (ordered + 1) // 2
        else:
            unordered = ordered // 2
        if unordered > best_count:
            second_best_N = best_N
            second_best_count = best_count
            best_N = N
            best_count = unordered
        elif unordered > second_best_count:
            second_best_N = N
            second_best_count = unordered
    
    print(f"Best N = {best_N}, count = {best_count}")
    print(f"Second best N = {second_best_N}, count = {second_best_count}")
    
    # Generate pairs for best_N
    pairs = []
    for a in pw_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in pw_set:
            pairs.append([a, b])
    
    pairs.sort()
    
    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.1f}s, found {len(pairs)} pairs for N={best_N}")
    
    return {'N': best_N, 'pairs': pairs}
