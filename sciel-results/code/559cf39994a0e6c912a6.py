"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80  # 80s safety margin within 90s limit
    
    N = 1000001  # need up to 1000000+1 for n+1
    
    # Build smallest prime factor sieve
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute powerful part for each number
    # powerful_part[x] = product of p^e for all prime powers p^e || x with e >= 2
    pow_part = [1] * (N + 1)
    
    for x in range(2, N + 1):
        if spf[x] == x:  # x is prime, powerful part is 1
            continue
        # Factor x using spf
        temp = x
        pp = 1
        while temp > 1:
            p = spf[temp]
            cnt = 0
            while temp % p == 0:
                temp //= p
                cnt += 1
            if cnt >= 2:
                pp *= (p ** cnt)
        pow_part[x] = pp
    
    # Find best n in [10000, 999999] maximizing pow_part[n] * pow_part[n+1]
    best_n = 10000
    best_val = pow_part[10000] * pow_part[10001]
    
    for n in range(10000, 1000000):
        val = pow_part[n] * pow_part[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
