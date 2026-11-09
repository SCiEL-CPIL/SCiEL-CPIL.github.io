import time
import math

def construct():
    start_time = time.time()
    MAX_N = 1000000
    MIN_N = 10000
    N_LIMIT = MAX_N + 1
    
    # 1. Compute Smallest Prime Factor (SPF)
    # spf[i] will hold the smallest prime factor of i
    spf = [0] * (N_LIMIT + 1)
    for i in range(2, N_LIMIT + 1):
        if spf[i] == 0:
            # i is a prime
            for j in range(i, N_LIMIT + 1, i):
                if spf[j] == 0:
                    spf[j] = i

    # 2. Compute powerful part for each number
    # p_part[i] will hold the powerful part of i
    p_part = [1] * (N_LIMIT + 1)
    
    # We can compute this efficiently.
    # For each i, factorize using spf.
    for i in range(2, N_LIMIT + 1):
        x = i
        res = 1
        while x > 1:
            p = spf[x]
            e = 0
            # Count exponent
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                res *= (p ** e)
        p_part[i] = res

    # 3. Search for max product
    best_n = MIN_N
    max_score = 0
    
    for n in range(MIN_N, MAX_N + 1):
        # Check if we are running out of time
        if time.time() - start_time > 80:
            break
            
        score = p_part[n] * p_part[n+1]
        if score > max_score:
            max_score = score
            best_n = n
            
    return best_n
