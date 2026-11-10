import time

def construct():
    deadline = time.time() + 55  # safety margin before 60s kill
    N = 123457
    
    # Build smallest prime factor (SPF) sieve up to N
    # Using a list of zeros is faster than range initialization for large N in some Python versions
    # But range is fine. Let's stick to standard.
    spf = list(range(N + 1))
    
    # Optimize the sieve loop
    # Standard SPF sieve
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute P(n) = largest prime factor of n, for all n in [0, N]
    P = [0] * (N + 1)
    P[0] = 0
    P[1] = 1
    for n in range(2, N + 1):
        p = spf[n]
        rest = n // p
        P[n] = max(p, P[rest])
    
    # Search for optimal a in [1, N-1] minimizing max(P(a), P(N-a))
    best_val = float('inf')
    best_a = 1
    
    # Local variables for speed
    P_local = P
    N_local = N
    
    for a in range(1, N_local):
        b = N_local - a
        val = P_local[a]
        pb = P_local[b]
        if pb > val:
            val = pb
        
        if val < best_val:
            best_val = val
            best_a = a
            
        # Time check every 10000 iterations to avoid overhead
        if a % 10000 == 0 and time.time() > deadline:
            break
    
    return [best_a, N - best_a]
