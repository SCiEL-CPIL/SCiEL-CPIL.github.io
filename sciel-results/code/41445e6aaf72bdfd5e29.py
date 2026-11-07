import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 140  # 10s margin from 150s limit
    max_M = 10**18

    # Strategy: Sieve-based sigma computation
    # Compute sigma(k) for all k up to N using a sieve, then group by M = k*sigma(k)
    # This is much faster than recursive smooth number generation and allows
    # checking far more candidates.

    # Choose N based on available time. Start with a moderate value and increase
    # if time permits.
    N = 50_000_000  # 50 million

    # Sieve for sigma: sigma(n) = sum of divisors of n
    # Use a modified sieve where for each prime p, we add p^a * (1 + 1/p + ... + 1/p^a)
    # Actually, simpler: initialize sigma = [1] * (N+1), then for each i from 1 to N,
    # add i to all multiples of i. But that's O(N log N) which might be too slow for N=50M.
    
    # Better: use the multiplicative property. For each prime p, for each power p^a,
    # update sigma for numbers divisible by p^a.
    
    # Most efficient: 
    # 1. Compute smallest prime factor (SPF) sieve
    # 2. Use SPF to factor each number and compute sigma
    # Or: use a direct sieve approach
    
    # Let's use a direct sieve: sigma[i] = sum of divisors of i
    # Initialize sigma = list(range(1, N+1))  # sigma[i] >= i (since i divides i)
    # Then for j from 1 to N, for each multiple m of j, add j to sigma[m]
    # But this is O(N * (1 + 1/2 + 1/3 + ... + 1/N)) = O(N log N)
    # For N = 50M, this is about 50M * 17 ≈ 850M operations, which is too slow in Python.
    
    # Alternative: use the formula sigma(n) = product of (p^(a+1)-1)/(p-1) for n = product p^a
    # We can compute this using SPF sieve.
    
    # Step 1: SPF sieve
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Step 2: Compute sigma for each number using SPF
    # sigma(n) = product over prime factors p^a || n of (p^(a+1)-1)/(p-1)
    sigma = [0] * (N + 1)
    sigma[1] = 1
    
    # For each n from 2 to N, factor using SPF and compute sigma
    # But factoring each number individually is O(log n) per number, total O(N log N)
    
    # Better: use dynamic programming with SPF
    # For each n, let p = spf[n], and let n = p^a * m where p doesn't divide m
    # Then sigma(n) = sigma(m) * (p^(a+1)-1)/(p-1)
    
    # Precompute for each n: the highest power of spf[n] dividing n, and the remaining part
    # Let's use arrays:
    #   exp[n] = exponent of spf[n] in n
    #   rem[n] = n / spf[n]^exp[n]
    #   sigma[n] = sigma[rem[n]] * (p^(exp[n]+1)-1)/(p-1) where p = spf[n]
    
    exp_arr = [0] * (N + 1)
    rem_arr = [1] * (N + 1)
    
    for n in range(2, N + 1):
        p = spf[n]
        if n % (p * p) == 0:
            # n is divisible by p^2
            exp_arr[n] = exp_arr[n // p] + 1
            rem_arr[n] = rem_arr[n // p]
        else:
            exp_arr[n] = 1
            rem_arr[n] = n // p
    
        # Compute sigma[n] = sigma[rem_arr[n]] * (p^(exp_arr[n]+1) - 1) // (p - 1)
        a = exp_arr[n]
        # Compute p^(a+1) - 1
        p_pow = p ** (a + 1)
        sigma_factor = (p_pow - 1) // (p - 1)
        sigma[n] = sigma[rem_arr[n]] * sigma_factor
    
    # Check time
    if time.time() > deadline - 20:
        # Not enough time, return a known good result
        return {'M': int(333312), 'k': [336, 372, 434]}
    
    # Step 3: Group by M = k * sigma(k)
    m_to_k_list = defaultdict(list)
    
    for k in range(2, N + 1):
        M = k * sigma[k]
        if M <= max_M:
            m_to_k_list[M].append(k)
    
    # Check time
    if time.time() > deadline - 5:
        # Find best from what we have so far
        best_M = 333312
        best_k_list = [336, 372, 434]
        best_count = 3
        for M, k_list in m_to_k_list.items():
            if len(k_list) > best_count:
                best_count = len(k_list)
                best_M = M
                best_k_list = k_list
        return {'M': int(best_M), 'k': [int(k) for k in best_k_list]}
    
    # Find best M
    best_M = 333312
    best_k_list = [336, 372, 434]
    best_count = 3
    
    for M, k_list in m_to_k_list.items():
        if len(k_list) > best_count:
            best_count = len(k_list)
            best_M = M
            best_k_list = k_list
    
    # Ensure types
    return {'M': int(best_M), 'k': [int(k) for k in best_k_list]}

if __name__ == "__main__":
    construct()
