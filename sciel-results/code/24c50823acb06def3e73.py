import time

def construct():
    deadline = time.time() + 80
    sp = (2, 3, 5, 7, 11, 13, 17, 19, 23)
    
    lo = 1001
    hi = 1000024
    size = hi - lo + 1
    
    # Sieve to compute small-prime parts for all numbers in [lo, hi]
    spp_arr = [1] * size
    
    for p in sp:
        first = ((lo + p - 1) // p) * p
        for m in range(first, hi + 1, p):
            idx = m - lo
            count = 0
            tmp = m
            while tmp % p == 0:
                count += 1
                tmp //= p
            spp_arr[idx] *= p ** count
    
    best_n = 1000
    best_c = 99
    
    for n in range(1000, 1000001):
        if time.time() > deadline:
            break
        start = n + 1 - lo
        c = len(set(spp_arr[start:start+24]))
        if c < best_c:
            best_c = c
            best_n = n
            if c == 2:
                return n
    
    return best_n
