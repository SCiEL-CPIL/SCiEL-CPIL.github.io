import math

def construct():
    n = 255
    # Find prime factors of n
    factors = set()
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            factors.add(d)
            while temp % d == 0:
                temp //= d
        d += 1
    if temp > 1:
        factors.add(temp)
    
    best_set = []
    for p in factors:
        # Generate multiples of p in [1, n]
        multiples = list(range(p, n + 1, p))
        if len(multiples) > len(best_set):
            best_set = multiples
            
    return best_set
