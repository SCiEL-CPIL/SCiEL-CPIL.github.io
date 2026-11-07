import math
import time

def construct():
    deadline = time.time() + 140
    N = 10**7 # Try 10 million
    
    # Generate SPF
    spf = list(range(N+1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i: # i is prime
            for j in range(i*i, N+1, i):
                if spf[j] == j:
                    spf[j] = i
    spf[0] = 0
    spf[1] = 1
    
    sol = {}
    best_M = 0
    best_ks = []
    
    # Helper to compute sigma from spf
    # We can optimize this by memoizing or just calculating.
    # Since we iterate k sequentially, we can't easily memoize sigma(k) for non-prime factors without a table.
    # But we can compute sigma(k) in O(log k).
    
    for k in range(1, N+1):
        if time.time() > deadline - 5:
            break
            
        # Compute sigma(k)
        if k == 1:
            s = 1
        else:
            s = 1
            temp = k
            while temp > 1:
                p = spf[temp]
                exp = 0
                term = 1
                while temp % p == 0:
                    temp //= p
                    exp += 1
                    term = term * p + 1
                s *= term
        M = k * s
        if M > 10**18:
            continue # or break if M is increasing? 
            # M is not strictly increasing, but generally is.
            # k^2 <= M. If k^2 > 10^18, break.
            # k <= 10^9.
            
        if M in sol:
            sol[M].append(k)
            if len(sol[M]) > len(best_ks):
                best_ks = sol[M]
                best_M = M
        else:
            sol[M] = [k]
            
    return {'M': best_M, 'k': best_ks}
