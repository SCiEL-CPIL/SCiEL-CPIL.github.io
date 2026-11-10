import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (150s limit, 5s buffer)
    
    # Use a linear sieve to compute sigma efficiently in O(N)
    # sigma is multiplicative: if n = p1^a1 * p2^a2 * ... then
    # sigma(n) = sigma(p1^a1) * sigma(p2^a2) * ...
    # where sigma(p^a) = (p^(a+1) - 1) / (p - 1)
    
    # We'll use a modified linear sieve that tracks:
    # - sigma[n]: sum of divisors of n
    # - lp[n]: largest prime factor of n (or we can use spf)
    # - p_pow[n]: the highest power of the smallest prime factor dividing n
    
    # Actually, let's use the standard approach:
    # sigma[1] = 1
    # For each n, if n is prime: sigma[n] = 1 + n
    # If n = m * p where p is the smallest prime factor of n and p divides m:
    #   n = p^a * r where gcd(p,r) = 1
    #   sigma[n] = sigma[r] * (1 + p + ... + p^a)
    # We need to track the p-part separately.
    
    # Linear sieve approach:
    # primes = []
    # sigma = [0] * (N+1)
    # sigma[1] = 1
    # spf = [0] * (N+1)  # smallest prime factor
    # p_pow = [0] * (N+1)  # highest power of spf dividing n
    # sigma_p = [0] * (N+1)  # sigma of the p-part (highest power of spf)
    
    # For n from 2 to N:
    #   if spf[n] == 0:  # n is prime
    #     primes.append(n)
    #     spf[n] = n
    #     sigma[n] = 1 + n
    #     p_pow[n] = n
    #     sigma_p[n] = 1 + n
    #   for p in primes:
    #     if p > spf[n] or n * p > N:
    #       break
    #     spf[n*p] = p
    #     if p == spf[n]:
    #       # n = p^a * r, so n*p = p^(a+1) * r
    #       p_pow[n*p] = p_pow[n] * p
    #       sigma_p[n*p] = sigma_p[n] + p_pow[n*p]
    #       sigma[n*p] = sigma[n] // sigma_p[n] * sigma_p[n*p]
    #       break
    #     else:
    #       # p < spf[n], so gcd(p, n) = 1
    #       p_pow[n*p] = p
    #       sigma_p[n*p] = 1 + p
    #       sigma[n*p] = sigma[n] * (1 + p)
    
    # This is O(N) time.
    
    # Let's choose N based on available time and memory.
    # Memory: sigma, spf, p_pow, sigma_p, is_prime (if needed) - each ~8 bytes per entry
    # For N = 10^8: 5 arrays * 10^8 * 8 bytes = 4 GB - too much
    # For N = 5*10^7: 5 arrays * 5*10^7 * 8 bytes = 2 GB - still a lot
    # For N = 2*10^7: 5 arrays * 2*10^7 * 8 bytes = 800 MB - might be ok
    # For N = 10^7: 5 arrays * 10^7 * 8 bytes = 400 MB - safe
    
    # Let's start with N = 10^7 and see if we can increase it.
    N = 10**7
    
    # Check time estimate: O(N) operations, each simple. Should take ~1-2 seconds for N=10^7.
    
    primes = []
    sigma = [0] * (N + 1)
    spf = [0] * (N + 1)
    p_pow = [0] * (N + 1)
    sigma_p = [0] * (N + 1)
    
    sigma[1] = 1
    spf[1] = 1
    
    for n in range(2, N + 1):
        if time.time() > deadline - 30:
            break
        if spf[n] == 0:
            primes.append(n)
            spf[n] = n
            sigma[n] = 1 + n
            p_pow[n] = n
            sigma_p[n] = 1 + n
        for p in primes:
            if p > spf[n] or n * p > N:
                break
            np_ = n * p
            spf[np_] = p
            if p == spf[n]:
                p_pow[np_] = p_pow[n] * p
                sigma_p[np_] = sigma_p[n] + p_pow[np_]
                sigma[np_] = sigma[n] // sigma_p[n] * sigma_p[np_]
                break
            else:
                p_pow[np_] = p
                sigma_p[np_] = 1 + p
                sigma[np_] = sigma[n] * (1 + p)
    
    # Now find M with most solutions k where k * sigma[k] = M
    # Use a dictionary to count occurrences of each M
    # M = k * sigma[k], and we need M <= 10^18
    
    M_limit = 10**18
    m_count = defaultdict(int)
    m_ks = defaultdict(list)
    
    for k in range(1, N + 1):
        if time.time() > deadline - 15:
            break
        M = k * sigma[k]
        if M <= M_limit:
            m_count[M] += 1
            if len(m_ks[M]) < 1000:  # Cap at 1000 to save memory
                m_ks[M].append(k)
    
    # Find M with maximum count
    best_M = 0
    best_count = 0
    for M, count in m_count.items():
        if count > best_count:
            best_count = count
            best_M = M
    
    if best_M == 0:
        # Fallback to a known collision
        return {'M': 333312, 'k': [336, 372, 434]}
    
    best_ks = m_ks[best_M]
    
    # Verify and return
    if len(best_ks) > 0:
        return {'M': int(best_M), 'k': [int(k) for k in best_ks]}
    else:
        return {'M': 333312, 'k': [336, 372, 434]}
