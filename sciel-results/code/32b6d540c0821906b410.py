import math
import time

def factorize(n):
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def get_divisors(factors):
    divs = [1]
    for p, exp in factors.items():
        new_divs = []
        p_power = 1
        for e in range(exp + 1):
            for d in divs:
                new_divs.append(d * p_power)
            p_power *= p
        divs = new_divs
    return divs

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        # Try x from n to n+20
        for x in range(n, n + 21):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
                
            # We need d | B^2
            # (Ay-B)(Az-B) = B^2
            # Let d = Ay-B, e = Az-B
            # y = (d+B)/A, z = (e+B)/A
            # Conditions:
            # 1. d | B^2
            # 2. (d+B) % A == 0
            # 3. (e+B) % A == 0
            # 4. y, z distinct and distinct from x
            # 5. z <= 10^12
            
            B_sq = B * B
            # Factorize B_sq
            # Optimization: If B is large, factorization is fast.
            fac_B_sq = factorize(B_sq)
            divisors = get_divisors(fac_B_sq)
            
            # Sort divisors? We want to find a valid one quickly.
            # Maybe sort by magnitude? 
            # Small d -> large e -> large z.
            # Large d -> small e -> small z.
            # We want z <= 10^12.
            # If z is too large, try larger d.
            # So we should check divisors in increasing order of d?
            # Or just check all. Number of divisors is small (< 1000).
            
            for d in divisors:
                if (d + B) % A != 0:
                    continue
                e = B_sq // d
                if (e + B) % A != 0:
                    continue
                
                y = (d + B) // A
                z = (e + B) // A
                
                if y < 1 or z < 1:
                    continue
                
                if x == y or y == z or x == z:
                    continue
                    
                if z > 10**12:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
