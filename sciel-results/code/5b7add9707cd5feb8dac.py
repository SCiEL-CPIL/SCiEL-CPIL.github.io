"""Rewritten program: find M <= 10^18 with many distinct k <= sqrt(M) satisfying k*sigma(k) = M."""
import time
import math
from collections import defaultdict

# EVOLVE-BLOCK-START

def _isqrt(x):
    return math.isqrt(x)

def _sigma_by_factorization(n):
    """Compute sigma(n) by trial division factorization."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    result = 1
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            power = 0
            while temp % d == 0:
                temp //= d
                power += 1
            factor = (d ** (power + 1) - 1) // (d - 1)
            result *= factor
        d += 1
    if temp > 1:
        result *= (1 + temp)
    return result

def _divisors(n):
    """Return sorted list of all divisors of n."""
    if n <= 0:
        return []
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    divs.sort()
    return divs

def _linear_sigma_sieve(N):
    """Compute sigma(k) for k=1..N using a linear sieve. Returns list indexed 0..N."""
    sigma = [0] * (N + 1)
    sigma[1] = 1
    primes = []
    is_composite = [False] * (N + 1)
    # For the linear sieve for sigma, we need:
    # sigma is multiplicative. For prime p: sigma(p) = 1+p.
    # For p^a: sigma(p^a) = (p^(a+1)-1)/(p-1)
    # We track: for each n, the largest prime power p^a dividing n (where p is the smallest prime factor),
    # and the product of sigma of the rest.
    # Standard linear sieve for sigma:
    # Let spf[n] = smallest prime factor.
    # Let p_power[n] = the highest power of spf[n] dividing n.
    # Let sigma_rest[n] = sigma(n / p_power[n]).
    # Then sigma(n) = sigma(p_power[n]) * sigma_rest[n].
    
    # Actually, a cleaner approach:
    # sigma[n] for n=1 is 1.
    # For composite n = p * m where p = spf[n]:
    #   If p does not divide m: sigma[n] = sigma[p] * sigma[m] = (1+p) * sigma[m]
    #   If p divides m: let m = p^a * r where p does not divide r.
    #     Then n = p^(a+1) * r.
    #     sigma[n] = sigma[p^(a+1)] * sigma[r]
    #     We need to track the p-power part.
    
    # Let's use arrays:
    # spf[n]: smallest prime factor
    # p_pow[n]: the largest power of spf[n] that divides n
    # sigma_p_pow[n]: sigma(p_pow[n])
    # rest_sigma[n]: sigma(n / p_pow[n])
    
    spf = [0] * (N + 1)
    p_pow = [1] * (N + 1)  # p_pow[1] = 1
    sigma_p = [1] * (N + 1)  # sigma of p_pow part
    rest_sig = [1] * (N + 1)  # sigma of the rest
    
    sigma[1] = 1
    p_pow[1] = 1
    sigma_p[1] = 1
    rest_sig[1] = 1
    
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
            p_pow[i] = i
            sigma_p[i] = 1 + i
            rest_sig[i] = 1
            sigma[i] = 1 + i
        for p in primes:
            if p > spf[i] or i * p > N:
                break
            spf[i * p] = p
            if p == spf[i]:
                # p divides i, so p^a || i for some a >= 1
                # p_pow[i] = p^a, now p_pow[i*p] = p^(a+1) = p_pow[i] * p
                p_pow[i * p] = p_pow[i] * p
                sigma_p[i * p] = sigma_p[i] + p_pow[i * p]  # sigma(p^(a+1)) = sigma(p^a) + p^(a+1)
                rest_sig[i * p] = rest_sig[i]
            else:
                # p does not divide i (and p < spf[i] is impossible since we break when p > spf[i])
                # Actually since p <= spf[i] and p != spf[i], we have p < spf[i]
                # But wait, in linear sieve we iterate p over primes with p <= spf[i].
                # If p < spf[i], then p does not divide i.
                p_pow[i * p] = p
                sigma_p[i * p] = 1 + p
                rest_sig[i * p] = sigma[i]
            sigma[i * p] = sigma_p[i * p] * rest_sig[i * p]
    
    return sigma

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Phase 1: Sieve sigma up to N, find M with most k's
    # Start with a moderate N, increase if time allows
    N = 2000000
    sigma = _linear_sigma_sieve(N)
    
    m_to_k = defaultdict(list)
    for k in range(1, N + 1):
        M_val = k * sigma[k]
        if M_val <= 10**18:
            m_to_k[M_val].append(k)
    
    # Find top candidates by count
    candidates = sorted(m_to_k.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Keep top 100 candidates for extension
    top_candidates = candidates[:100]
    
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    # For each top candidate, try to extend with k > N
    for M_val, ks in top_candidates:
        if time.time() > deadline - 15:
            break
        
        sqrt_M = _isqrt(M_val)
        # All k in ks should satisfy k <= sqrt_M (verified during sieve since k <= N and we check)
        valid_ks = [k for k in ks if k <= sqrt_M]
        current_count = len(valid_ks)
        
        if current_count <= best_count and current_count + 2 <= best_count:
            # Even if we add a couple more, won't beat best
            # But let's be more generous: skip if clearly can't beat
            if current_count + 5 <= best_count:
                continue
        
        # Try to extend: find divisors k of M_val with k > N, k <= sqrt_M
        # where sigma(k) = M_val // k
        divs = _divisors(M_val)
        existing = set(valid_ks)
        extended = list(valid_ks)
        
        for k in divs:
            if k <= N or k > sqrt_M or k in existing:
                continue
            expected_sigma = M_val // k
            if expected_sigma < k:  # sigma(k) >= k+1 for k>1, so if expected_sigma < k+1 skip
                # Actually sigma(k) >= 1 + k for k > 1, so sigma(k) >= k+1
                # If k == 1, sigma(1) = 1. But k > N >= 2, so k >= 2.
                if expected_sigma < k + 1:
                    continue
            if _sigma_by_factorization(k) == expected_sigma:
                extended.append(k)
                existing.add(k)
        
        if len(extended) > best_count:
            best_M = M_val
            best_ks = sorted(extended)
            best_count = len(extended)
    
    # Phase 2: If time remains, try a larger sieve
    if time.time() < deadline - 25:
        N2 = 5000000
        sigma2 = _linear_sigma_sieve(N2)
        
        m_to_k2 = defaultdict(list)
        for k in range(1, N2 + 1):
            M_val = k * sigma2[k]
            if M_val <= 10**18:
                m_to_k2[M_val].append(k)
        
        candidates2 = sorted(m_to_k2.items(), key=lambda x: len(x[1]), reverse=True)
        top_candidates2 = candidates2[:100]
        
        for M_val, ks in top_candidates2:
            if time.time() > deadline - 10:
                break
            
            sqrt_M = _isqrt(M_val)
            valid_ks = [k for k in ks if k <= sqrt_M]
            current_count = len(valid_ks)
            
            if current_count + 5 <= best_count:
                continue
            
            divs = _divisors(M_val)
            existing = set(valid_ks)
            extended = list(valid_ks)
            
            for k in divs:
                if k <= N2 or k > sqrt_M or k in existing:
                    continue
                expected_sigma = M_val // k
                if k > 1 and expected_sigma < k + 1:
                    continue
                if _sigma_by_factorization(k) == expected_sigma:
                    extended.append(k)
                    existing.add(k)
            
            if len(extended) > best_count:
                best_M = M_val
                best_ks = sorted(extended)
                best_count = len(extended)
    
    # Ensure types are correct
    best_M = int(best_M)
    best_ks = [int(k) for k in best_ks]
    
    # Final validation: verify all k satisfy k*sigma(k) == M and k <= sqrt(M)
    sqrt_M = _isqrt(best_M)
    final_ks = []
    for k in best_ks:
        if k <= sqrt_M and _sigma_by_factorization(k) * k == best_M:
            final_ks.append(k)
    
    return {'M': best_M, 'k': final_ks}

# EVOLVE-BLOCK-END
