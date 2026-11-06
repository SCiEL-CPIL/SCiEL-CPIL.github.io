import time
import math

def construct():
    t0 = time.time()
    DEADLINE = 80  # safety margin before 90s kill
    
    def elapsed():
        return time.time() - t0
    
    limit = 1000000
    lower = 10000
    N = limit + 1  # need up to 1000001 for n+1 when n=1000000
    
    # Linear sieve to compute powerful parts
    spf = [0] * (N + 1)
    cnt = [0] * (N + 1)
    rest = [0] * (N + 1)
    pp = [1] * (N + 1)
    primes = []
    
    # Initialize 1
    spf[1] = 1
    cnt[1] = 0
    rest[1] = 1
    pp[1] = 1
    
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            cnt[i] = 1
            rest[i] = 1
            pp[i] = 1  # Prime itself has exponent 1, so powerful part is 1
            primes.append(i)
        
        for p in primes:
            if p > spf[i] or i * p > N:
                break
            
            spf[i * p] = p
            
            if p == spf[i]:
                cnt[i * p] = cnt[i] + 1
                rest[i * p] = rest[i]
                # cnt[i*p] is at least 2
                pp[i * p] = pp[rest[i * p]] * (p ** cnt[i * p])
            else:
                cnt[i * p] = 1
                rest[i * p] = i
                pp[i * p] = pp[i]
        
        if elapsed() > DEADLINE:
            break
    
    # Scan for best consecutive pair
    best_n = lower
    best_val = 0
    for n in range(lower, limit):
        v = pp[n] * pp[n + 1]
        if v > best_val:
            best_val = v
            best_n = n
    
    return best_n
