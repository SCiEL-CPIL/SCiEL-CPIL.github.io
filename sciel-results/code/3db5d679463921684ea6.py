import time
import math

def construct():
    deadline = time.time() + 85.0
    answers = []
    
    # Helper to factorize a number
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

    # Helper to generate divisors from factorization
    def get_divisors(factors):
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            for d in divs:
                for i in range(e + 1):
                    new_divs.append(d * (p ** i))
            divs = new_divs
        return divs

    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        # x range: n/4 < x < 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            A = 4 * x - n
            B = n * x
            
            # U * V = B^2
            # U = A*y - B
            # V = A*z - B
            # y > x => U > A*x - B = 2x(2x-n)
            # z > y => V > U
            # Also y, z distinct from x and each other.
            # If x < y < z, then distinctness is satisfied.
            
            lower_bound = 2 * x * (2 * x - n)
            
            # If lower_bound >= B, no solution for this x with y > x
            # Since U < V and U*V=B^2, U < B.
            if lower_bound >= B:
                continue
                
            # Factorize B^2
            # B = n*x
            # We can factorize B and double exponents
            factors_B = factorize(B)
            factors_B2 = {p: e * 2 for p, e in factors_B.items()}
            
            divs = get_divisors(factors_B2)
            
            for U in divs:
                if U >= B:
                    continue
                if U <= lower_bound:
                    continue
                # Check divisibility: (U + B) must be divisible by A
                # y = (U + B) / A
                if (U + B) % A != 0:
                    continue
                
                y = (U + B) // A
                V = (B * B) // U
                # Check z
                if (V + B) % A != 0:
                    continue
                z = (V + B) // A
                
                # Check ordering and distinctness
                if x < y < z:
                    found = [x, y, z]
                    break
            if found:
                break
                
        answers.append(found)
        
    return answers
