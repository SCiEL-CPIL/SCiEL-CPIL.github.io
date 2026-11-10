import math
import time

def construct():
    MAX_DENOM = 10**12
    DEADLINE = time.time() + 50  # Safety margin
    
    def factorize(num):
        factors = {}
        d = 2
        while d * d <= num:
            while num % d == 0:
                factors[d] = factors.get(d, 0) + 1
                num //= d
            d += 1
        if num > 1:
            factors[num] = factors.get(num, 1)
        return factors

    def get_divisors_less_than_b_squared(b, b_squared):
        # Returns list of divisors U of b_squared such that U < b
        # Optimized: just generate all divisors and filter
        factors = factorize(b)
        # Generate divisors of b_squared
        # b_squared factors have exponents 2*e
        base_primes = list(factors.keys())
        base_exps = [2 * factors[p] for p in base_primes]
        
        divisors = [1]
        for i, p in enumerate(base_primes):
            current_divs = divisors[:]
            power = 1
            for _ in range(base_exps[i]):
                power *= p
                for d in current_divs:
                    divisors.append(d * power)
            # divisors now contains divisors of p^exponent * previous
            # Actually, the standard way is:
            # new_divs = []
            # for d in divisors:
            #   for k in range(exp+1):
            #      new_divs.append(d * p^k)
            # divisors = new_divs
            
        # Let's re-implement divisor generation correctly
        divs = [1]
        for p, e in factors.items():
            exp_in_b2 = 2 * e
            new_divs = []
            for d in divs:
                mult = 1
                for _ in range(exp_in_b2 + 1):
                    new_divs.append(d * mult)
                    mult *= p
            divs = new_divs
            
        # Filter U < b
        valid_us = [u for u in divs if u < b]
        return valid_us

    answers = []
    for n in range(3, 257):
        if time.time() > DEADLINE:
            answers.append(None)
            continue
            
        found = None
        # x must be in (n/4, 3n/4)
        x_start = n // 4 + 1
        x_end = (3 * n) // 4
        
        for x in range(x_start, x_end + 1):
            if time.time() > DEADLINE:
                break
                
            A = 4 * x - n
            B = n * x
            
            # We need 1/y + 1/z = A/B
            # y, z > x
            # U = Ay - B. Since y > x, U > Ax - B.
            # Ax - B = x(4x-n) - nx = 4x^2 - 2nx.
            # Also U < B (since y < z implies U < V, and UV=B^2)
            
            # Check if A > 0
            if A <= 0:
                continue
                
            b_squared = B * B
            
            # Get divisors U of B^2 such that U < B
            # To save time, we can just iterate divisors of B^2.
            # But B can be up to 256 * 192 ~ 50000. B^2 ~ 2.5e9.
            # Factorization is fast.
            
            factors = factorize(B)
            divs = [1]
            for p, e in factors.items():
                exp_in_b2 = 2 * e
                new_divs = []
                for d in divs:
                    mult = 1
                    for _ in range(exp_in_b2 + 1):
                        new_divs.append(d * mult)
                        mult *= p
                divs = new_divs
                
            # We need U < B and (U + B) % A == 0
            # Also y = (U+B)/A must be integer > x
            # And z = (V+B)/A must be integer > y
            
            for U in divs:
                if U >= B:
                    continue
                    
                if (U + B) % A != 0:
                    continue
                    
                y = (U + B) // A
                if y <= x:
                    continue
                    
                V = b_squared // U
                if (V + B) % A != 0:
                    continue
                    
                z = (V + B) // A
                if z <= y:
                    continue
                
                if z <= MAX_DENOM:
                    found = [x, y, z]
                    break
                    
            if found:
                break
                
        answers.append(found)
        
    return answers
