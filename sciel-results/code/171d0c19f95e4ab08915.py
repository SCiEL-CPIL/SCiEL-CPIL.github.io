import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140
    max_m = 10**18

    # Find best K_MAX that fits in time
    # Sieve O(K log log K) + sigma O(K) + counting O(K)
    # For K=10^6: ~50ms. For K=5*10^6: ~250ms.
    # We'll try increasing K until time runs out.
    
    best_result = {'M': 333312, 'k': [336, 372, 434]}
    best_score = 3
    
    # Try several K values, keeping the best
    for K in [10**6, 2*10**6, 4*10**6, 8*10**6]:
        if time.time() > deadline - 20:
            break
        
        # Sieve smallest prime factor
        spf = list(range(K + 1))
        for i in range(2, int(K**0.5) + 1):
            if spf[i] == i:  # i is prime
                for j in range(i*i, K + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        
        # Compute sigma(k) for all k using spf
        sigma = [0] * (K + 1)
        sigma[1] = 1
        for k in range(2, K + 1):
            p = spf[k]
            m = k // p
            if m % p == 0:
                # p divides m, so k has p^a where a >= 2
                # sigma(k) = sigma(m) * p / (1 - p) ... no
                # If k = p^a * r, m = p^(a-1) * r
                # sigma(k) = sigma(r) * (1 + p + ... + p^a)
                # sigma(m) = sigma(r) * (1 + p + ... + p^(a-1))
                # sigma(k) = sigma(m) + p^a * sigma(r)
                # p^a = k // r = k * p / (m * p) ... hmm
                # Actually: k = p^a * r, m = p^(a-1) * r
                # p^a = p * m / r ... no, p^a = k/r, p^(a-1) = m/r
                # sigma(k) = sigma(m) + p^a * sigma(r)
                # But we don't have sigma(r) directly.
                # Alternative: sigma(k) = sigma(m) + k/r * sigma(r)
                # k/r = p^a, m/r = p^(a-1), so k/r = p * m/r = p * (m/r)
                # Hmm, let me use: if p^a || k, then sigma(k) = sigma(k/p^a) * (p^(a+1)-1)/(p-1)
                # But that's complex. Let me use the recurrence:
                # If k = p^a * r where p ∤ r:
                #   sigma(k) = sigma(r) * (1 + p + p^2 + ... + p^a)
                # If k = p^(a+1) * r where p ∤ r:
                #   sigma(k) = sigma(r) * (1 + p + ... + p^(a+1))
                #   = sigma(p^a * r) * (1 + p + ... + p^(a+1)) / (1 + p + ... + p^a)
                # This is getting complex. Let me just compute directly.
                
                # Simpler approach: use the formula
                # sigma(k) where k = p^a * r, p ∤ r
                # sigma(k) = sigma(r) * (p^(a+1) - 1) / (p - 1)
                # We need to find r and a.
                # Since spf[k] = p, we can factor out all p's.
                # But that's O(log k) per k, which is too slow.
                
                # Better: precompute using the multiplicative property
                # Actually, the standard approach:
                # sigma[1] = 1
                # For k > 1, let p = spf[k], m = k//p
                # If m % p == 0:
                #   Let p^a || k, so p^(a-1) || m
                #   sigma(k) = sigma(m) + p^a * sigma(k/p^a)
                #   But k/p^a is the part not divisible by p
                #   Alternatively: sigma(k) = sigma(m) + k * sigma(k/p^a) / p^a
                #   Hmm, this is still complex.
                
                # Let me just use a different approach: compute sigma via factorization
                # Actually, the cleanest way:
                # sigma is multiplicative. For k = prod p_i^a_i, sigma(k) = prod (p_i^(a_i+1)-1)/(p_i-1)
                # We can compute this using spf.
                
                # For each k, factor it using spf and compute sigma.
                # This is O(number of prime factors) per k, which averages O(log log k).
                # Total: O(K log log K), which is fine.
                
                # Actually, there's a cleaner recurrence:
                # If k = p * m where p = spf[k]:
                #   If m % p == 0:
                #     Let p^a || m (so p^(a+1) || k)
                #     sigma(k) = sigma(m) + p^(a+1) * sigma(m / p^a)
                #     But we don't store p^a or m/p^a.
                #     
                #   If m % p != 0:
                #     sigma(k) = sigma(m) * (1 + p)
                #
                # For the case m % p == 0, we need more info.
                # Let's store for each k: the highest power of spf[k] dividing k.
                # Or, let's just factor each k directly.
                
                # Direct factorization approach:
                temp = k
                s = 1
                while temp > 1:
                    p = spf[temp]
                    power = 1
                    while temp % p == 0:
                        temp //= p
                        power *= p
                    s *= (power * p - 1) // (p - 1)
                sigma[k] = s
            else:
                # m not divisible by p, so k = p * m with p ∤ m
                sigma[k] = sigma[m] * (1 + p)
        
        # Count solutions: for each k, M = k * sigma[k]
        # Count how many k give the same M
        M_to_ks = defaultdict(list)
        for k in range(1, K + 1):
            M = k * sigma[k]
            if M <= max_m:
                M_to_ks[M].append(k)
        
        # Find M with most solutions
        for M, ks in M_to_ks.items():
            if len(ks) > best_score:
                # Verify all ks satisfy k <= isqrt(M) and k*sigma(k) = M
                valid_ks = []
                for k in ks:
                    if k <= math.isqrt(M) and k * sigma[k] == M:
                        valid_ks.append(k)
                if len(valid_ks) > best_score:
                    best_score = len(valid_ks)
                    best_result = {'M': int(M), 'k': [int(k) for k in valid_ks]}
        
        if time.time() > deadline - 10:
            break
    
    # Safety: ensure types are correct
    result = best_result
    result['M'] = int(result['M'])
    result['k'] = [int(k) for k in result['k']]
    
    return result
