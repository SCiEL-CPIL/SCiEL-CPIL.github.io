import time
import math

def sieve_spf(limit):
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

MAX_N = 256
MAX_X = (3 * MAX_N) // 4
MAX_B = MAX_N * MAX_X
SPF = sieve_spf(MAX_B + 1)

def get_divisors_of_square(num):
    # Factorize num
    factors = {}
    temp = num
    while temp > 1:
        p = SPF[temp]
        count = 0
        while temp % p == 0:
            temp //= p
            count += 1
        factors[p] = count
    
    # Generate divisors of num^2
    # If num = p1^e1 * p2^e2, then num^2 = p1^(2e1) * p2^(2e2)
    # Divisors are p1^(k1) * p2^(k2) ... where 0 <= ki <= 2*ei
    
    divisors = [1]
    for p, e in factors.items():
        max_exp = 2 * e
        new_divs = []
        for d in divisors:
            for k in range(max_exp + 1):
                new_divs.append(d * (p ** k))
        divisors = new_divs
    return divisors

def construct():
    answers = []
    start_time = time.time()
    deadline = start_time + 85.0
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        min_x = n // 4 + 1
        max_x = (3 * n) // 4
        
        for x in range(min_x, max_x + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
                
            min_U = 4 * x * x - 2 * n * x + 1
            max_U = B - 1
            
            if min_U > max_U:
                continue
                
            # Get divisors of B^2
            # Optimization: Only check divisors in range
            # Generating all divisors might be faster than iterating range if range is large
            # B^2 can be large.
            
            divs = get_divisors_of_square(B)
            
            for U in divs:
                if U < min_U or U > max_U:
                    continue
                    
                if (U + B) % A != 0:
                    continue
                    
                y = (U + B) // A
                
                if y <= x:
                    continue
                    
                V = (B * B) // U
                
                if (V + B) % A != 0:
                    continue
                    
                z = (V + B) // A
                
                if z <= y:
                    continue
                
                # Check distinctness and bounds
                # x, y, z are positive integers
                # x < y < z ensures distinctness
                # Bounds <= 10^12
                if x <= 10**12 and y <= 10**12 and z <= 10**12:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
