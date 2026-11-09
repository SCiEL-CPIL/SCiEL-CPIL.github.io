"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    
    N = 1000001  # we need up to 1,000,001 to compute powerful_part for n and n+1 where n in [10000, 1000000]
    
    # Linear sieve to compute:
    # spf[m]  = smallest prime factor of m
    # exp[m]  = exponent of spf[m] in m
    # rest[m] = m / (spf[m] ** exp[m])
    
    spf = np.zeros(N + 1, dtype=np.int64)
    exp = np.zeros(N + 1, dtype=np.int64)
    rest = np.zeros(N + 1, dtype=np.int64)
    primes = []
    
    # Linear sieve
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            exp[i] = 1
            rest[i] = 1
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > N:
                break
            spf[i * p] = p
            if p == spf[i]:
                # p divides i, so exponent increases
                exp[i * p] = exp[i] + 1
                rest[i * p] = rest[i]
                break
            else:
                # p does not divide i
                exp[i * p] = 1
                rest[i * p] = i
    
    # Now compute powerful_part[m] using DP:
    # powerful_part[m] = (spf[m] ** exp[m]) * powerful_part[rest[m]] if exp[m] >= 2
    #                    = powerful_part[rest[m]] otherwise
    # Base case: powerful_part[1] = 1
    
    powerful = np.ones(N + 1, dtype=np.int64)
    
    for m in range(2, N + 1):
        r = rest[m]
        e = exp[m]
        if e >= 2:
            powerful[m] = (spf[m] ** e) * powerful[r]
        else:
            powerful[m] = powerful[r]
    
    # Now find n in [10000, 1000000] that maximizes powerful[n] * powerful[n+1]
    best_n = 10000
    best_val = powerful[10000] * powerful[10001]
    
    for n in range(10000, 1000000):
        val = powerful[n] * powerful[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    return int(best_n)
# EVOLVE-BLOCK-END
