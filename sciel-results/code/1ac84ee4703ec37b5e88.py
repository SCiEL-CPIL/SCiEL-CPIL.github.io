import time
from math import gcd

# Precompute primes up to a reasonable limit for factorization
# Max B approx 256 * 5 * 256 = 327680. sqrt(327680) approx 572.
# Let's go up to 1000 to be safe.
def _sieve(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

PRIMES = _sieve(1000)

def _factorize(n):
    """Return prime factorization of n as dict {prime: exponent}."""
    factors = {}
    if n <= 1:
        return factors
    temp_n = n
    for p in PRIMES:
        if p * p > temp_n:
            break
        while temp_n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            temp_n //= p
    if temp_n > 1:
        factors[temp_n] = factors.get(temp_n, 0) + 1
    return factors

def _all_divisors(factors):
    """Generate all divisors from prime factorization dict."""
    divs = [1]
    for p, e in factors.items():
        powers = [p ** i for i in range(e + 1)]
        divs = [d * pw for d in divs for pw in powers]
    return divs

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin for 90s limit
    BOUND = 10**12
    
    answers = []
    
    for n in range(3, 257):
        if time.time() - start_time > DEADLINE:
            answers.append(None)
            continue
        
        found = None
        
        # x must be > n/4. 
        # If we assume x < y < z, then x < 3n/4.
        # We will search x from n//4 + 1 to 5*n to be safe, 
        # but prioritize smaller x as they are more likely to yield small y,z.
        x_min = n // 4 + 1
        x_max = 5 * n
        
        for x in range(x_min, x_max + 1):
            if time.time() - start_time > DEADLINE:
                break
            
            a = 4 * x - n
            b = n * x
            
            if a <= 0:
                continue
            
            b_sq = b * b
            bf = _factorize(b)
            b_sq_factors = {p: 2 * e for p, e in bf.items()}
            divs = _all_divisors(b_sq_factors)
            
            for d in divs:
                if time.time() - start_time > DEADLINE:
                    break
                
                # We want (a*y - b) * (a*z - b) = b^2
                # Let u = a*y - b, v = a*z - b. u*v = b^2.
                # y = (u+b)/a, z = (v+b)/a.
                # To avoid duplicates and ensure y <= z, we can assume u <= v, so u <= b.
                if d > b:
                    continue
                
                num_y = d + b
                if num_y % a != 0:
                    continue
                y = num_y // a
                
                v = b_sq // d
                num_z = v + b
                if num_z % a != 0:
                    continue
                z = num_z // a
                
                if y <= 0 or z <= 0:
                    continue
                if y > BOUND or z > BOUND:
                    continue
                
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                
                # We found a solution. 
                # We don't strictly need x < y < z for validity, just distinctness.
                # But finding one with x as smallest is efficient.
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
