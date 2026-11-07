import time
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # Strategy: enumerate k up to some bound, compute M = k*sigma(k),
    # group by M, find M with most k values.
    
    # First, let's do a smart enumeration.
    # We need k * sigma(k) = M <= 10^18, so k * sigma(k) <= 10^18.
    # sigma(k) >= k+1, so k^2 + k <= 10^18, roughly k <= 10^9.
    # But we can't enumerate up to 10^9.
    
    # Instead, use the structure: if k = prod p_i^e_i, then sigma(k) = prod (p_i^(e_i+1)-1)/(p_i-1).
    # M = k * sigma(k) = prod p_i^(e_i+1) * (p_i-1) * prod (p_i^(e_i+1)-1)/(p_i-1) ... no.
    # M = prod p_i^e_i * prod (1 + p_i + ... + p_i^e_i)
    #   = prod p_i^e_i * prod (p_i^(e_i+1)-1)/(p_i-1)
    
    # Alternative approach: for each M, the number of solutions k is the number of divisors k of M
    # such that k*sigma(k)=M. Note k <= sqrt(M) since sigma(k) >= k+1 > k for k>=1.
    # Actually sigma(k) >= k+1 for k>=1, so k*(k+1) <= M, k <= (sqrt(4M+1)-1)/2.
    
    # Let's try a different structural approach:
    # Enumerate "smooth" structures of k. For k with few prime factors,
    # we can enumerate the exponents and compute M, then group.
    
    # Key insight from the seed: k=336=2^4*3*7, k=372=2^2*3*31, k=434=2*7*31
    # M=333312 = 2^6 * 3 * 7 * 31 * ... let's check.
    # 333312 / 64 = 5208, 5208/3=1736, 1736/7=248, 248/31=8. So 333312=2^6*3*7*31. Wait 64*3*7*31=64*651=41664. No.
    # 333312 = 2^5 * 3 * 7 * 31 * ... 32*3*7*31=20832. 333312/20832=16. So 2^9*3*7*31? 512*651=333312. Yes!
    
    # Let's enumerate k by its prime factorization pattern.
    # For each small set of primes and exponent combinations, compute k and M=k*sigma(k).
    
    # Phase 1: Enumerate k up to K_LIMIT by direct computation
    K_LIMIT = 200000
    M_to_ks = defaultdict(list)
    
    for k in range(1, K_LIMIT + 1):
        # Compute sigma(k)
        # For speed, use a sieve-like approach
        pass
    
    # Better: use a sieve to compute sigma for all k up to K_LIMIT
    def sigma_sieve(n):
        """Compute sigma(k) for all k up to n using a sieve."""
        sigma = [0] * (n + 1)
        for d in range(1, n + 1):
            for m in range(d, n + 1, d):
                sigma[m] += d
        return sigma
    
    sigma_arr = sigma_sieve(K_LIMIT)
    
    for k in range(1, K_LIMIT + 1):
        M = k * sigma_arr[k]
        if M <= 10**18:
            M_to_ks[M].append(k)
    
    # Find the best M
    for M, ks in M_to_ks.items():
        count = len(ks)
        if count > best_count:
            best_count = count
            best_M = M
            best_k = ks[:]
    
    # Check deadline
    if time.time() > deadline - 20:
        # Return what we have
        ks = sorted(best_k)
        return {'M': int(best_M), 'k': [int(x) for x in ks[:1000]]}
    
    # Phase 2: Extend to larger k using smarter enumeration.
    # For k > K_LIMIT, we can't do a full sieve easily.
    # But we can enumerate k by their prime factorization.
    # k = p1^a1 * p2^a2 * ... with small primes.
    
    # Let's also try: for M up to some range, enumerate divisors k of M where k <= sqrt(M)
    # and check if k*sigma(k) = M. This is expensive.
    
    # Instead, let's enumerate more k values using a different sieve range.
    # Actually, let's try going up to 10^6 for sigma sieve.
    K_LIMIT2 = 1000000
    if time.time() < deadline - 30:
        sigma_arr2 = sigma_sieve(K_LIMIT2)
        for k in range(K_LIMIT + 1, K_LIMIT2 + 1):
            M = k * sigma_arr2[k]
            if M <= 10**18:
                M_to_ks[M].append(k)
        
        # Re-find best
        for M, ks in M_to_ks.items():
            count = len(ks)
            if count > best_count:
                best_count = count
                best_M = M
                best_k = ks[:]
    
    # Phase 3: Try to find even more solutions by targeting specific M structures.
    # For M with many prime factors, there may be more k solutions.
    # Let's try to extend our search by enumerating k with specific prime structures.
    
    # Actually, let's just make sure we return the best we found.
    ks = sorted(best_k)
    # Ensure all k satisfy k <= isqrt(M)
    isqrt_M = math.isqrt(best_M)
    ks = [k for k in ks if k <= isqrt_M]
    
    return {'M': int(best_M), 'k': [int(x) for x in ks[:1000]]}
# EVOLVE-BLOCK-END
