import math

def construct():
    M = 1000000
    # We need P(n) for n up to M, and P(n+1) for n up to M, so up to M+1
    limit = M + 1
    
    # Sieve of Eratosthenes to find smallest prime factor
    spf = [0] * (limit + 1)
    for i in range(2, limit + 1):
        if spf[i] == 0:
            spf[i] = i
            if i * i <= limit:
                for j in range(i * i, limit + 1, i):
                    if spf[j] == 0:
                        spf[j] = i
    
    # Calculate powerful part for each number
    # P(x) is product of p^e where e>=2 in x's factorization
    powerful_part = [1] * (limit + 1)
    
    for i in range(2, limit + 1):
        x = i
        pp = 1
        while x > 1:
            p = spf[x]
            count = 1
            x //= p
            while x > 1 and spf[x] == p:
                x //= p
                count += 1
            if count >= 2:
                pp *= (p ** count)
        powerful_part[i] = pp
        
    # Find n that maximizes P(n) * P(n+1)
    max_val = 0
    best_n = 10000
    
    for n in range(10000, M + 1):
        val = powerful_part[n] * powerful_part[n+1]
        if val > max_val:
            max_val = val
            best_n = n
            
    return best_n
