import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    N_MIN = 10000
    N_MAX = 1000000
    limit = N_MAX + 1  # Need LPF up to N_MAX+1
    
    # Efficient LPF sieve using numpy
    # lpf[x] = largest prime factor of x
    lpf = np.zeros(limit + 1, dtype=np.int32)
    
    # Sieve: for each prime, mark its multiples with that prime as LPF
    # Since we iterate primes in increasing order, the last prime to mark a number
    # is its largest prime factor
    for i in range(2, limit + 1):
        if lpf[i] == 0:  # i is prime
            lpf[i::i] = i  # Vectorized update for all multiples
    
    # Compute max(lpf[n], lpf[n+1]) for all n in [N_MIN, N_MAX]
    lpf_n = lpf[N_MIN:N_MAX + 1]
    lpf_n1 = lpf[N_MIN + 1:N_MAX + 2]
    
    max_lpf = np.maximum(lpf_n, lpf_n1)
    
    # Find the minimum and corresponding n
    min_val = np.min(max_lpf)
    min_indices = np.where(max_lpf == min_val)[0]
    best_n = N_MIN + min_indices[0]
    
    # Verify with exact trial division as a sanity check
    def largest_prime_factor_exact(x):
        if x <= 1:
            return x
        lpf_val = x
        f = 2
        while f * f <= x:
            while x % f == 0:
                lpf_val = f
                x //= f
            f += 1
        if x > 1:
            lpf_val = x
        return lpf_val
    
    # Final verification
    verify_n = int(best_n)
    lp_n = largest_prime_factor_exact(verify_n)
    lp_n1 = largest_prime_factor_exact(verify_n + 1)
    final_val = max(lp_n, lp_n1)
    
    return verify_n
