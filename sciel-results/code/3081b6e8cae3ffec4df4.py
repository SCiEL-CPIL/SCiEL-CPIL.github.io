import math

def construct():
    M = 1000000
    limit = M + 1
    
    # Linear Sieve to compute P(n) for all n up to limit
    # P(n) is the powerful part (product of p^e for e>=2)
    
    P = [1] * (limit + 1)
    exp = [0] * (limit + 1) # exponent of smallest prime factor
    rest = [1] * (limit + 1) # n divided by spf(n)^exp(n)
    is_composite = [False] * (limit + 1)
    primes = []
    
    P[1] = 1
    exp[1] = 0
    rest[1] = 1
    
    for i in range(2, limit + 1):
        if not is_composite[i]:
            primes.append(i)
            P[i] = 1
            exp[i] = 1
            rest[i] = 1
        
        for p in primes:
            if i * p > limit:
                break
            is_composite[i * p] = True
            
            if i % p == 0:
                # p is the smallest prime factor of i
                exp[i * p] = exp[i] + 1
                rest[i * p] = rest[i]
                
                if exp[i] == 1:
                    # i = p * m, P(i) = P(m)
                    # i*p = p^2 * m, P(i*p) = p^2 * P(m) = p^2 * P(i)
                    P[i * p] = P[i] * p * p
                else:
                    # i = p^k * m, k>=2, P(i) = p^k * P(m)
                    # i*p = p^(k+1) * m, P(i*p) = p^(k+1) * P(m) = p * P(i)
                    P[i * p] = P[i] * p
                break
            else:
                # p < spf(i), so spf(i*p) = p
                exp[i * p] = 1
                rest[i * p] = i
                # P(i*p) = P(i) * P(p) = P(i) * 1
                P[i * p] = P[i]
                
    # Find n that maximizes P(n) * P(n+1)
    max_val = 0
    best_n = 10000
    
    for n in range(10000, M + 1):
        val = P[n] * P[n + 1]
        if val > max_val:
            max_val = val
            best_n = n
            
    return best_n
