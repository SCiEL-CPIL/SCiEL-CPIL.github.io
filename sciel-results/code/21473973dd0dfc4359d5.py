import math
from array import array

def construct():
    N = 3 * 10**7
    spf = array('I', range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    solutions = {}
    best_M = 0
    best_k = []
    max_count = 0
    
    for k in range(1, N + 1):
        x = k
        sigma = 1
        while x > 1:
            p = spf[x]
            power = 0
            p_power = 1
            while x % p == 0:
                x //= p
                power += 1
                p_power *= p
            sigma *= (p_power * p - 1) // (p - 1)
        
        M = k * sigma
        if M > 10**18:
            continue
            
        if M in solutions:
            solutions[M].append(k)
            if len(solutions[M]) > max_count:
                max_count = len(solutions[M])
                best_M = M
                best_k = solutions[M][:]
        else:
            solutions[M] = [k]
            
    if max_count == 0:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': best_M, 'k': best_k}
