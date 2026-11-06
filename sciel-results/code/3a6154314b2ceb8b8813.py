"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START

def _build_spf(limit):
    """Build smallest prime factor sieve up to limit."""
    spf = list(range(limit + 1))
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _powerful_part(n, spf):
    """Product of p^e for all p^e || n with e >= 2, using SPF table."""
    if n <= 1:
        return 1
    pp = 1
    temp = n
    while temp > 1:
        p = spf[temp]
        e = 0
        while temp % p == 0:
            temp //= p
            e += 1
        if e >= 2:
            pp *= p ** e
    return pp

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin before 90s kill
    
    # Build SPF up to 1,000,000
    max_n = 1000000
    spf = _build_spf(max_n)
    
    best_n = 10000
    best_val = -1
    
    # Scan all valid n in [10000, 999999] (n+1 must be <= 1000000)
    for n in range(10000, max_n):
        if time.time() > deadline:
            break
        
        pp_n = _powerful_part(n, spf)
        if pp_n == 1:
            continue  # skip if n has no powerful part (all exponents are 1)
        
        pp_np1 = _powerful_part(n + 1, spf)
        if pp_np1 == 1:
            continue
        
        val = pp_n * pp_np1
        if val > best_val:
            best_val = val
            best_n = n
    
    return best_n

# EVOLVE-BLOCK-END
