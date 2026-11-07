"""Rewritten program: compute k*sigma(k) for k up to a limit, bucket collisions, return best M."""
import math
import time
from collections import defaultdict

def _sigma(n):
    """Compute sigma(n) = sum of divisors of n using trial division up to sqrt(n)."""
    if n == 1:
        return 1
    result = 1
    temp = n
    # Factor 2
    count = 0
    while temp % 2 == 0:
        count += 1
        temp //= 2
    if count:
        result *= (2**(count+1) - 1)
    # Odd factors
    f = 3
    while f * f <= temp:
        if temp % f == 0:
            count = 0
            while temp % f == 0:
                count += 1
                temp //= f
            result *= (f**(count+1) - 1) // (f - 1)
        f += 2
    if temp > 1:
        result *= (temp + 1)
    return result

def construct():
    deadline = time.time() + 130  # 130s to leave margin before 150s kill
    
    # We'll compute k*sigma(k) for k in ranges and look for collisions.
    # Strategy: process k in chunks, maintaining a dictionary M -> list of k's.
    # To keep memory manageable, we process in chunks and only keep M values
    # that have seen at least one k (potential collisions).
    
    # Actually, let's think about this differently.
    # k*sigma(k) is roughly k^2 for k that are prime (sigma(k)=k+1), so M ~ k^2.
    # For k up to 10^9, M can be up to ~10^18.
    # But we need k <= sqrt(M), which is automatically satisfied if M = k*sigma(k) since sigma(k) >= k+1 > k for k>=1... wait no.
    # sigma(k) >= k+1 only for prime k. For composite k, sigma(k) can be less than or greater than k.
    # Actually sigma(k) >= k+1 always (since 1 and k are divisors, and for k>1, sigma(k) >= 1+k).
    # So M = k*sigma(k) >= k*(k+1) > k^2, so k < sqrt(M). Good, k <= isqrt(M) is always satisfied.
    
    # So we just need to find M values that are hit by multiple distinct k values.
    # Let's compute k*sigma(k) for k from 1 up to some limit, and find the M with the most hits.
    
    # Memory concern: if we go up to k = 10^7, that's 10^7 entries. Each entry is (M, k).
    # M is up to ~10^14 for k=10^7. Let's see how far we can go.
    
    # Let's use a two-pass approach:
    # Pass 1: compute all (M, k) pairs, store in a dict M -> [k1, k2, ...]
    # But this could use too much memory for large ranges.
    
    # Alternative: process in chunks. For each chunk, compute M values and check against
    # previously seen M values. But we need to accumulate all k's for each M.
    
    # Let's just do it directly with a dict, and see how far we get in time.
    # k*sigma(k) computation: for each k, we need sigma(k) which takes O(sqrt(k)) worst case.
    # For k up to 10^6, sqrt(k) ~ 1000, so 10^6 * 1000 = 10^9 operations... too slow.
    
    # Better: use a sieve-like approach to compute sigma for all k up to N.
    # sigma can be computed using a modified sieve in O(N log N).
    
    N = 10**6  # Start with 1 million
    # Check if we can afford this in time
    # Sieve for sigma: O(N log N) ~ 10^6 * 14 = 1.4 * 10^7 operations, very fast.
    
    # Let's compute sigma for all k up to N using a sieve.
    sigma_arr = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma_arr[j] += i
    
    # Now compute M = k * sigma(k) for each k, and bucket by M.
    buckets = defaultdict(list)
    for k in range(1, N + 1):
        M = k * sigma_arr[k]
        if M <= 10**18:
            buckets[M].append(k)
    
    # Find the M with the most k's
    best_M = 0
    best_k_list = []
    for M, k_list in buckets.items():
        if len(k_list) > len(best_k_list):
            best_M = M
            best_k_list = k_list
    
    # If we have time, try larger N
    elapsed = time.time() - (deadline - 130)
    if elapsed < 100 and N < 10**7:
        # Extend to N = 10^7
        N2 = 10**7
        sigma_arr2 = [0] * (N2 + 1)
        for i in range(1, N2 + 1):
            for j in range(i, N2 + 1, i):
                sigma_arr2[j] += i
        
        # Clear old buckets and recompute
        buckets.clear()
        for k in range(1, N2 + 1):
            M = k * sigma_arr2[k]
            if M <= 10**18:
                buckets[M].append(k)
        
        best_M = 0
        best_k_list = []
        for M, k_list in buckets.items():
            if len(k_list) > len(best_k_list):
                best_M = M
                best_k_list = k_list
    
    # Fallback
    if not best_k_list:
        best_M = 333312
        best_k_list = [336, 372, 434]
    
    # Truncate to 1000 max
    best_k_list = best_k_list[:1000]
    
    return {'M': int(best_M), 'k': [int(x) for x in best_k_list]}
