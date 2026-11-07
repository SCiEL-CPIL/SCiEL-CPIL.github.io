"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from collections import defaultdict

# EVOLVE-BLOCK-START

def _sigma(n):
    """Compute sigma(n) = sum of divisors of n."""
    if n == 1:
        return 1
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            result *= (p ** (exp + 1) - 1) // (p - 1)
        p += 1 if p == 2 else 2
    if temp > 1:
        result *= (temp * temp - 1) // (temp - 1)
    return result

def _find_solutions_for_M(M):
    """Find all k <= sqrt(M) such that k * sigma(k) == M."""
    if M < 1:
        return []
    limit = math.isqrt(M)
    sols = []
    for k in range(1, limit + 1):
        if M % k == 0:
            if k * _sigma(k) == M:
                sols.append(k)
    return sols

def _search_best_M():
    """Search for M with many solutions to k*sigma(k)=M."""
    deadline = time.time() + 140  # safety margin before 150s kill
    best_M = 333312
    best_sols = [336, 372, 434]
    
    # Strategy: try M values that are highly composite or have special structure
    # k*sigma(k) = M means M/k = sigma(k), so M = k * sigma(k)
    # We want many k with the same product k*sigma(k)
    
    # Approach: generate many k values, compute M_k = k*sigma(k), 
    # then find M values that appear most frequently
    
    from collections import Counter
    M_counts = Counter()
    M_to_ks = defaultdict(list)
    
    # Generate k values up to some limit and compute k*sigma(k)
    # We need k <= sqrt(M) and M <= 10^18
    # So k*sigma(k) <= 10^18, meaning k <= 10^18/sigma(k)
    # For small k, sigma(k) is small, so k can be large
    # But we need k <= sqrt(M) = sqrt(k*sigma(k)), so k^2 <= k*sigma(k), k <= sigma(k)
    # This is always true for k >= 1 since sigma(k) >= k+1 for k > 1, and sigma(1)=1
    # Wait: k <= sigma(k) is always true. So the constraint k <= sqrt(M) is:
    # k <= sqrt(k*sigma(k)) => k^2 <= k*sigma(k) => k <= sigma(k) which is always true.
    # So the constraint is automatically satisfied!
    
    # So we just need to find M values that are k*sigma(k) for many k.
    # Let's scan k from 1 upward and count how many times each M = k*sigma(k) appears.
    
    max_k_to_scan = 200000  # tune based on time
    for k in range(1, max_k_to_scan + 1):
        if time.time() > deadline - 5:
            break
        s = _sigma(k)
        M_val = k * s
        if M_val <= 10**18:
            M_counts[M_val] += 1
            M_to_ks[M_val].append(k)
    
    # Find the M with the most solutions
    if M_counts:
        max_count = max(M_counts.values())
        candidates = [m for m, c in M_counts.items() if c == max_count]
        # Pick the smallest M among ties (or any)
        best_M_candidate = min(candidates)
        best_sols_candidate = M_to_ks[best_M_candidate]
        
        if max_count > len(best_sols):
            best_M = best_M_candidate
            best_sols = best_sols_candidate
    
    return best_M, best_sols

def construct():
    M, sols = _search_best_M()
    # Verify and ensure valid types
    M = int(M)
    sols = [int(x) for x in sols]
    # Ensure distinct
    sols = list(dict.fromkeys(sols))
    # Ensure k_i <= isqrt(M)
    limit = math.isqrt(M)
    sols = [k for k in sols if 1 <= k <= limit]
    # Ensure k*sigma(k) == M
    sols = [k for k in sols if k * _sigma(k) == M]
    # Ensure M <= 10^18
    if M > 10**18:
        M = 333312
        sols = [336, 372, 434]
    if not sols:
        M = 333312
        sols = [336, 372, 434]
    return {'M': M, 'k': sols}
# EVOLVE-BLOCK-END
