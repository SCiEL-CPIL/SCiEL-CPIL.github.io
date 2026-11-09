"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
from collections import defaultdict
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Precompute sigma values using sieve
    # We need k such that k*sigma(k) = M for multiple k values
    # Strategy: find M with many representations
    
    # First, let's try a constructive approach
    # For M = 2^a * 3^b * 5^c * ... we can look for k values
    
    # Let's use a different approach: search for M values that have many k solutions
    # k * sigma(k) = M means sigma(k) = M/k, so k must divide M
    
    # Approach: For small k, compute k*sigma(k) and find collisions
    # This is O(N log N) with a sieve
    
    # Let's compute k*sigma(k) for k up to some limit and find M with many preimages
    # But we need M <= 10^18 and k <= sqrt(M)
    
    # Key insight: if k * sigma(k) = M, then k <= sqrt(M) means sigma(k) >= sqrt(M)
    # So sigma(k) >= k, which is always true (sigma(k) >= k+1 for k > 1)
    
    # Let's try: compute f(k) = k*sigma(k) for k in a range, find M with many k values
    # such that all k <= sqrt(M)
    
    # For k up to N, f(k) can be up to N * sigma(N) ~ N^2 * log(log(N))
    # So M ~ N^2, and we need k <= sqrt(M) ~ N, which is satisfied
    
    # Let's compute f(k) for k up to N and find the M with most preimages
    # where all preimages k satisfy k <= sqrt(M)
    
    # Since f(k) is roughly k^2 * (some factor), for a given M, 
    # k <= sqrt(M) is approximately k <= sqrt(f(k)/sigma(k)*sigma(k)) = k
    # Actually k <= sqrt(M) = sqrt(k*sigma(k)) means k^2 <= k*sigma(k) means k <= sigma(k)
    # which is always true for k >= 1
    
    # So the constraint k <= sqrt(M) is automatically satisfied!
    
    # Let's just find M with the most preimages k where f(k) = M
    
    # Use a sieve to compute sigma(k) for k up to N
    N = 200000  # Start with a reasonable limit
    
    # Sieve for sigma
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
    
    # Now compute f(k) = k * sigma(k) and count preimages
    f_count = defaultdict(list)
    for k in range(1, N + 1):
        m = k * sigma[k]
        if m <= 10**18:
            f_count[m].append(k)
    
    # Find the M with the most preimages
    best_m = None
    best_k_list = []
    for m, k_list in f_count.items():
        if len(k_list) > len(best_k_list):
            best_m = m
            best_k_list = k_list
    
    # Check if we found something good
    if best_m is not None and len(best_k_list) >= 3:
        # Verify all k <= sqrt(M)
        sqrt_m = math.isqrt(best_m)
        valid_k = [k for k in best_k_list if k <= sqrt_m]
        if len(valid_k) >= 3:
            return {'M': best_m, 'k': valid_k}
    
    # If the sieve approach didn't work well, try a constructive approach
    # Let's try to construct M with many solutions using number theory
    
    # For highly composite-like structures, we can try:
    # M = LCM of several k*sigma(k) values
    
    # Alternative: use known results about multiperfect numbers
    # A number n is multiperfect if sigma(n)/n is an integer
    # If n is multiperfect with sigma(n)/n = r, then n*sigma(n) = r*n^2
    
    # Let's try another approach: search for M by trying to find many k
    # such that k*sigma(k) are all equal
    
    # This is hard, so let's fall back to the best we found
    if best_m is not None:
        sqrt_m = math.isqrt(best_m)
        valid_k = [k for k in best_k_list if k <= sqrt_m]
        if valid_k:
            return {'M': best_m, 'k': valid_k}
    
    # Fallback to the original solution
    return {'M': 333312, 'k': [336, 372, 434]}
# EVOLVE-BLOCK-END
