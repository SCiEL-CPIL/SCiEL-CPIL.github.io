import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Known good baseline
    best_M = 333312
    best_ks = {336, 372, 434}
    
    # Search for better M by iterating k
    # We want to find M with many k such that k*sigma(k)=M
    # Group by M
    m_to_ks = {}
    
    # Limit k to a reasonable range to fit in time
    # sigma(k) <= k log log k approx. M <= k^2 log log k.
    # If k=10^6, M ~ 10^13.
    # If k=10^7, M ~ 10^15.
    # 10^18 allows k up to ~10^9, but that's too slow for naive sigma.
    # We will search k up to 2,000,000.
    
    max_k = 2000000
    
    # Precompute smallest prime factor (SPF) for fast sigma calculation?
    # Or just use trial division for each k? Trial division up to sqrt(k) is slow for 2M k's.
    # Let's use a sieve for smallest prime factor.
    
    def compute_spf(n):
        spf = list(range(n + 1))
        for i in range(2, int(n**0.5) + 1):
            if spf[i] == i:
                for j in range(i*i, n + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        return spf
    
    spf = compute_spf(max_k)
    
    def sigma(n):
        if n == 1:
            return 1
        result = 1
        while n > 1:
            p = spf[n]
            # Calculate sum of powers of p
            power_sum = 1
            power_val = 1
            while n % p == 0:
                n //= p
                power_val *= p
                power_sum += power_val
            result *= power_sum
        return result
    
    for k in range(1, max_k + 1):
        if time.time() > deadline:
            break
        s = sigma(k)
        m = k * s
        if m > 10**18:
            break # k is increasing, sigma(k) is increasing, so m is increasing
        if m not in m_to_ks:
            m_to_ks[m] = []
        m_to_ks[m].append(k)
        
        # Update best if we found more solutions
        if len(m_to_ks[m]) > len(best_ks):
            best_M = m
            best_ks = set(m_to_ks[m])
            
    # If we found a better M, return it
    if len(best_ks) > 3:
        return {'M': int(best_M), 'k': sorted([int(x) for x in best_ks])}
    else:
        return {'M': 333312, 'k': [336, 372, 434]}
