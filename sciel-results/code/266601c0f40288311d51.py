import time
from functools import lru_cache

@lru_cache(maxsize=None)
def factorize(n):
    """
    Returns a list of (prime, exponent) tuples for n.
    """
    if n <= 1:
        return []
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            count = 0
            while n % d == 0:
                n //= d
                count += 1
            factors.append((d, count))
        d += 1
    if n > 1:
        factors.append((n, 1))
    return tuple(factors)

def get_divisors_of_square(factors):
    """
    Given the prime factorization of B, returns a list of all divisors of B^2.
    If B = prod p_i^e_i, then B^2 = prod p_i^(2e_i).
    """
    divisors = [1]
    for p, e in factors:
        max_exp = 2 * e
        new_divs = []
        for d in divisors:
            p_pow = 1
            for k in range(max_exp + 1):
                new_divs.append(d * p_pow)
                p_pow *= p
        divisors = new_divs
    return divisors

def construct():
    deadline = time.time() + 55.0  # Safety margin
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            # Fill remaining with None if we run out of time
            while len(answers) < 254:
                answers.append(None)
            return answers
            
        found = None
        # x must satisfy n/4 < x < 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
            
            B = n * x
            
            # We need (Ay - B)(Az - B) = B^2
            # Let u = Ay - B, v = Az - B
            # u * v = B^2
            # y = (u + B) / A
            # z = (v + B) / A
            # Conditions:
            # 1. u, v are integers
            # 2. y, z are integers => u % A == -B % A and v % A == -B % A
            # 3. x < y < z
            #    y > x => (u + B) / A > x => u + B > Ax => u > Ax - B
            #    z > y => v > u (since z,y > 0 and A > 0) => v > u
            #    Since uv = B^2, v > u implies u < B
            
            # Pre-compute divisors of B^2
            fac_b = factorize(B)
            divs_b2 = get_divisors_of_square(fac_b)
            
            target_mod = (-B) % A
            # Lower bound for u to ensure y > x
            u_lower = (A * x - B) + 1 
            
            for u in divs_b2:
                if u >= B:
                    continue
                if u <= u_lower:
                    continue
                if u % A != target_mod:
                    continue
                
                # u is a candidate
                v = (B * B) // u
                # Check if v is also congruent to -B mod A
                if v % A != target_mod:
                    continue
                
                y = (u + B) // A
                z = (v + B) // A
                
                # Verify distinctness and order
                # u < B implies v > u implies z > y
                # u > u_lower implies y > x
                if y > x and z > y:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
