"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    answers = [None] * 254  # Index 0 corresponds to n=3, index 253 to n=256
    
    # Precompute primes up to a limit sufficient for factorizing B = n*x
    # Max n = 256, max x < n, so B < n^2 < 65536. Sieve to 100000 for safety.
    limit = 100000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    
    # Cache for factorizations to avoid repeated work
    factor_cache = {}
    
    def factorize(num):
        if num in factor_cache:
            return factor_cache[num]
        if num <= 1:
            factor_cache[num] = []
            return []
        factors = []
        temp = num
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                factors.append((p, count))
        if temp > 1:
            factors.append((temp, 1))
        factor_cache[num] = factors
        return factors

    def get_divisors_from_factorization(factors):
        divisors = [1]
        for p, exp in factors:
            new_divs = []
            p_power = 1
            for _ in range(exp + 1):
                for d in divisors:
                    new_divs.append(d * p_power)
                p_power *= p
            divisors = new_divs
        divisors.sort()
        return divisors

    for idx in range(254):
        n = idx + 3
        
        # Check time
        if time.time() > deadline:
            break
            
        found = None
        
        # We want 1/x + 1/y + 1/z = 4/n with x < y < z
        # Rearranging: (Ay - B)(Az - B) = B^2 where A = 4x - n, B = nx
        # u = Ay - B, v = Az - B, uv = B^2
        # y = (u+B)/A, z = (v+B)/A must be integers
        # x < y < z
        
        # x range: n/4 < x < 3n/4 (from 1/x > 4/(3n) since x is smallest)
        x_start = n // 4 + 1
        x_end = (3 * n + 2) // 4  # x < 3n/4 strictly
        
        for x in range(x_start, x_end + 1):
            # Time check in inner loop for large n
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
                
            # Factor B^2 by factoring B and doubling exponents
            b_factors = factorize(B)
            b_sq_factors = [(p, 2*e) for p, e in b_factors]
            
            divisors = get_divisors_from_factorization(b_sq_factors)
            
            # For each divisor u of B^2 with u <= B (so u <= v):
            # v = B^2 // u
            # Check divisibility and ordering
            for u in divisors:
                if u > B:
                    break
                v = (B * B) // u
                
                num_y = u + B
                if num_y % A != 0:
                    continue
                y = num_y // A
                
                num_z = v + B
                if num_z % A != 0:
                    continue
                z = num_z // A
                
                if x < y < z:
                    found = [x, y, z]
                    break
            
            if found:
                break
                
        answers[idx] = found
        
    return answers
# EVOLVE-BLOCK-END
