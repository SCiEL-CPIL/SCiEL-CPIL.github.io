import math

def construct():
    n = 255
    # Find prime factors of n
    factors = []
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            factors.append(d)
            while temp % d == 0:
                temp //= d
        d += 1
    if temp > 1:
        factors.append(temp)
    
    best_set = []
    max_size = 0
    
    for p in factors:
        # Generate multiples of p up to n
        count = n // p
        if count > max_size:
            max_size = count
            # We just need to return the list. 
            # For p=3, this is 85 elements.
            best_set = list(range(p, n + 1, p))
            
    return best_set
