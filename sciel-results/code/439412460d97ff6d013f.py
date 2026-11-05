import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill

    N = 1000000
    START = 10000

    # 1. Sieve for Smallest Prime Factor (SPF)
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # 2. Compute powerful part for each number using a multiplicative sieve approach
    # P[k] stores the powerful part of k (product of p^e for e>=2)
    P = [1] * (N + 1)
    
    # For each prime p, for each power p^e (e>=2), multiply P[multiples of p^e] by p^e
    # But we need to be careful: P[n] = product of p^e for each prime p dividing n with exponent e>=2
    # So for each prime p and each e>=2, for each multiple of p^e, we multiply by p^e
    # However, if p^e divides n, we should multiply by p^e exactly once (the full power)
    
    # Alternative: use the SPF to compute P[k] efficiently
    # P[k] can be computed from P[k//p] if we know the exponent of p in k
    # Let's use a different approach: for each k, factor using SPF and compute P[k]
    
    # Actually, let's use the multiplicative sieve:
    # Initialize P with 1
    # For each prime p:
    #   For e = 2, 3, ... while p^e <= N:
    #     For each multiple m of p^e:
    #       P[m] *= p^e
    # But this would overcount if p^e and p^(e+1) both divide m.
    # We want: for each prime p, if the exponent of p in m is e_p >= 2, multiply by p^e_p.
    
    # Better approach: for each prime p, for each number m, determine the exact exponent of p in m.
    # Use SPF-based computation:
    
    P = [1] * (N + 1)
    
    for k in range(2, N + 1):
        x = k
        p_part = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                p_part *= (p ** e)
        P[k] = p_part
    
    # 3. Find n in [10000, N-1] maximizing P[n] * P[n+1]
    best_n = START
    best_val = P[START] * P[START + 1]
    
    for n in range(START, N):
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        
        if time.time() > deadline:
            break
    
    return best_n
