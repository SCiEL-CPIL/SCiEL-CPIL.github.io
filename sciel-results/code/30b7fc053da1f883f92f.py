import time
import math

def _sieve_primes(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, val in enumerate(is_prime) if val]

_PRIMES = _sieve_primes(1000)

def _factorize(num):
    factors = {}
    temp = num
    for p in _PRIMES:
        if p * p > temp:
            break
        while temp % p == 0:
            factors[p] = factors.get(p, 0) + 1
            temp //= p
    if temp > 1:
        factors[temp] = factors.get(temp, 0) + 1
    return factors

def _get_divisors(factors):
    divisors = [1]
    for p, exp in factors.items():
        new_divs = []
        for d in divisors:
            for e in range(exp + 1):
                new_divs.append(d * (p ** e))
        divisors = new_divs
    return divisors

def construct():
    start_time = time.time()
    deadline = start_time + 85
    answers = []
    MAX_Z = 10**12
    n_total = 254
    
    for idx, n in enumerate(range(3, 257)):
        elapsed = time.time() - start_time
        remaining = deadline - start_time - elapsed
        remaining_n = n_total - idx
        
        if remaining <= 0:
            answers.append(None)
            continue
            
        found = None
        x_max = (3 * n) // 4
        x_min = n // 4 + 1
        
        if x_max < x_min:
            x_max = x_min
            
        n_factors = _factorize(n)
        
        for x in range(x_max, x_min - 1, -1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
                
            B = n * x
            
            # Factor B
            # B = n * x
            # We have n_factors. We need x_factors.
            x_factors = _factorize(x)
            
            # Merge factors
            B_factors = dict(n_factors)
            for p, e in x_factors.items():
                B_factors[p] = B_factors.get(p, 0) + e
                
            # We need divisors of B^2
            # So exponents are doubled
            B2_factors = {p: 2*e for p, e in B_factors.items()}
            
            divisors = _get_divisors(B2_factors)
            
            # Check divisors
            # D must be < B (since y < 2B/A => D < B)
            # And D > 0
            # Also D+B must be divisible by A
            # And B^2/D + B must be divisible by A
            
            for D in divisors:
                if D >= B:
                    continue
                    
                if (D + B) % A != 0:
                    continue
                    
                y = (D + B) // A
                
                # Check y > x
                if y <= x:
                    continue
                
                # Check z
                # z = (B^2/D + B) / A
                # B^2/D is integer since D divides B^2
                term = (B * B) // D + B
                if term % A != 0:
                    continue
                    
                z = term // A
                
                # Check distinctness and bounds
                if z <= y:
                    continue
                if z > MAX_Z:
                    continue
                
                # Valid solution
                found = [x, y, z]
                break
            
            if found:
                break
                
        answers.append(found)
        
    return answers
