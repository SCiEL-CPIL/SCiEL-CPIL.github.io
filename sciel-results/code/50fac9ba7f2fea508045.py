import time
import math

def _factorize(num):
    factors = {}
    d = 2
    while d * d <= num:
        while num % d == 0:
            factors[d] = factors.get(d, 0) + 1
            num //= d
        d += 1 if d == 2 else 2
    if num > 1:
        factors[num] = factors.get(num, 0) + 1
    return factors

def _get_divisors(factors):
    divisors = [1]
    for p, e in factors.items():
        new_divs = []
        for d in divisors:
            pe = 1
            for _ in range(e + 1):
                new_divs.append(d * pe)
                pe *= p
        divisors = new_divs
    return divisors

def construct():
    start_time = time.time()
    deadline = start_time + 50
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            while len(answers) < 254:
                answers.append(None)
            return answers
        
        found = None
        # x must be > n/4
        # Try x in a reasonable range
        # For most n, small x works.
        # Upper bound for x: if x is too large, K is large, N is large.
        # But we stop at first solution.
        # Let's try x from n//4 + 1 to n//4 + 1000
        # If not found, expand range?
        # Actually, for 4/n, solutions are very common.
        
        x_start = n // 4 + 1
        x_end = n // 4 + 10000
        
        for x in range(x_start, x_end):
            K = 4 * x - n
            if K <= 0:
                continue
            
            # N = (n*x)^2
            # We need divisors U of N such that U % K == (-n*x) % K
            # And V = N/U % K == (-n*x) % K
            
            nx = n * x
            N = nx * nx
            
            # Factorize N by factorizing nx and doubling exponents
            factors_nx = _factorize(nx)
            factors_N = {}
            for p, e in factors_nx.items():
                factors_N[p] = e * 2
            
            divisors = _get_divisors(factors_N)
            
            target_mod = (-nx) % K
            
            for U in divisors:
                if U % K != target_mod:
                    continue
                
                V = N // U
                if V % K != target_mod:
                    continue
                
                # Check if y, z are integers
                num_y = U + nx
                num_z = V + nx
                
                if num_y % K != 0 or num_z % K != 0:
                    continue
                
                y = num_y // K
                z = num_z // K
                
                if y <= 0 or z <= 0:
                    continue
                
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                
                # Check bound
                if y <= 10**12 and z <= 10**12:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
