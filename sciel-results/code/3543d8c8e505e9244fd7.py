# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N_MIN = 10000
    N_MAX = 1000000
    SIZE = N_MAX + 1
    
    # Sieve for smallest prime factor is not needed if we use the sf_1 sieve
    # We need to identify primes first
    is_prime = [True] * SIZE
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(SIZE)) + 1):
        if is_prime[i]:
            for j in range(i*i, SIZE, i):
                is_prime[j] = False
    
    # Compute sf_1(n): product of primes p such that v_p(n) == 1
    sf_1 = [1] * SIZE
    for p in range(2, SIZE):
        if is_prime[p]:
            p2 = p * p
            # Iterate multiples of p
            # If m is multiple of p but not p2, then v_p(m) == 1
            # So sf_1[m] should include p
            for m in range(p, SIZE, p):
                if m % p2 != 0:
                    sf_1[m] *= p
                # else: v_p(m) >= 2, so p is NOT in sf_1[m]
                
    # Compute pp(n) = n / sf_1(n)
    # We only care about [N_MIN, N_MAX]
    pp = [0] * SIZE
    for m in range(N_MIN, N_MAX + 1):
        pp[m] = m // sf_1[m]
        
    # Find best consecutive pair using NumPy
    pp_arr = np.array(pp[N_MIN:N_MAX+1], dtype=np.int64)
    # Products of consecutive elements
    products = pp_arr[:-1] * pp_arr[1:]
    
    if len(products) == 0:
        return N_MIN
        
    max_idx = np.argmax(products)
    best_val = products[max_idx]
    best_n = N_MIN + max_idx
    
    return best_n
# EVOLVE-BLOCK-END
