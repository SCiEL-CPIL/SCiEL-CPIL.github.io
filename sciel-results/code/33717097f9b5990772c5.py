import math
from collections import defaultdict

def construct():
    """
    Find M <= 10^18 with many distinct k <= sqrt(M) such that k * sigma(k) = M.
    """
    # Strategy:
    # 1. Precompute sigma(k) for k up to K_max using a sieve.
    # 2. For each k, compute M = k * sigma(k). If M <= 10^18 and k <= sqrt(M),
    #    add k to the list for M.
    # 3. Return the M with the most valid k's.
    
    # We need k <= sqrt(M) => k^2 <= M = k*sigma(k) => k <= sigma(k).
    # This is always true for k >= 1 (sigma(k) >= 1 + k > k for k > 1, sigma(1)=1).
    # Wait, sigma(k) >= k+1 for k>1, so k < sigma(k) is always true.
    # Actually, sigma(k) >= k+1 for k>1. For k=1, sigma(1)=1, so k=sigma(k).
    # So k <= sigma(k) is always true. Thus k <= sqrt(M) is equivalent to k^2 <= k*sigma(k) => k <= sigma(k), which is always true.
    # So the constraint k <= sqrt(M) is automatically satisfied for all k where M = k*sigma(k) is defined.
    # Wait, let's double check. M = k * sigma(k). We need k <= sqrt(M) => k^2 <= M => k^2 <= k * sigma(k) => k <= sigma(k).
    # Yes, this is always true. So we just need to count how many k map to the same M.
    
    # To get many solutions, we want M to have many divisors k such that sigma(k) = M/k.
    # This is a hard problem to solve directly. Instead, we search over k.
    
    # How large can k be? M <= 10^18. k * sigma(k) <= 10^18.
    # sigma(k) is roughly O(k log log k). So k^2 log log k <= 10^18 => k <= ~10^9 / sqrt(log log k).
    # So k can be up to around 10^8 or so. That's too many to sieve naively in Python.
    
    # Let's think differently. We want to maximize the number of k's for a fixed M.
    # Known highly composite numbers for this function?
    # Let's try to find M that is a multiple of many k*sigma(k) values.
    
    # Alternative approach: Generate candidate M values from small k, then verify.
    # But we need the one with the MOST solutions.
    
    # Let's use a smarter sieve. We can compute sigma(k) for k up to 10^6 or 10^7 in Python
    # within time limits if we're careful. 10^6 is definitely fine. 10^7 might be tight.
    # Let's try 10^6 first. If the best score is low, we can try to be smarter.
    
    # Actually, the seed program scores 1.6667. Let's see what it finds.
    # It uses N=1000000. Let's try to improve by either:
    # 1. Increasing N (risky for time).
    # 2. Being smarter about which M to pick.
    
    # Let's try N = 2000000 and see if it fits in time.
    # Sieve for sigma up to 2e6: sum_{i=1}^{2e6} 2e6/i ≈ 2e6 * ln(2e6) ≈ 2e6 * 14.5 ≈ 29e6 operations.
    # This should be fast enough in Python (maybe 1-2 seconds).
    
    N = 2000000
    MAX_M = 10**18
    
    # Sieve for sigma
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
    
    # Map M -> list of k
    m_to_ks = defaultdict(list)
    
    for k in range(1, N + 1):
        M_val = k * sigma[k]
        if M_val <= MAX_M:
            m_to_ks[M_val].append(k)
    
    # Find M with most k's
    best_M = 0
    best_k = []
    
    for M_val, ks in m_to_ks.items():
        if len(ks) > len(best_k):
            best_k = ks
            best_M = M_val
    
    if best_M == 0:
        # Fallback
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Verify: ensure all k_i are distinct and k_i <= sqrt(M)
    # We already know k_i are distinct (each k is processed once).
    # We know k_i <= sqrt(M) is always true as argued above.
    # But let's double-check for safety.
    sqrt_M = math.isqrt(best_M)
    valid_ks = [k for k in best_k if k <= sqrt_M]
    
    if not valid_ks:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    return {'M': best_M, 'k': valid_ks}
