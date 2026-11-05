import math
import time

def construct():
    N = 1000000
    MIN_N = 10000
    LIMIT = N + 1
    
    start_time = time.time()
    deadline = start_time + 55

    # 1. Sieve for Smallest Prime Factor (SPF)
    spf = list(range(LIMIT + 1))
    isqrt_limit = int(math.isqrt(LIMIT))
    for i in range(2, isqrt_limit + 1):
        if spf[i] == i:
            for j in range(i * i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # 2. Precompute exponent of smallest prime factor and remaining part
    exp = [0] * (LIMIT + 1)
    rem = [1] * (LIMIT + 1)
    
    for i in range(2, LIMIT + 1):
        if spf[i] == i:
            exp[i] = 1
            rem[i] = 1
        else:
            p = spf[i]
            k = i // p
            if spf[k] == p:
                exp[i] = exp[k] + 1
                rem[i] = rem[k]
            else:
                exp[i] = 1
                rem[i] = k

    if time.time() > deadline:
        return MIN_N

    # 3. Precompute powerful part pp[i]
    pp = [1] * (LIMIT + 1)
    pp[1] = 1
    
    for i in range(2, LIMIT + 1):
        if exp[i] >= 2:
            p = spf[i]
            pp[i] = pp[rem[i]] * (p ** exp[i])
        else:
            pp[i] = pp[rem[i]]

    if time.time() > deadline:
        return MIN_N

    # 4. Find max product
    best_n = MIN_N
    best_val = 0
    
    for n in range(MIN_N, N + 1):
        if time.time() > deadline:
            break
        val = pp[n] * pp[n+1]
        if val > best_val:
            best_val = val
            best_n = n

    return best_n
