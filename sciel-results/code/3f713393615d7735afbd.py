import time
from math import gcd

# EVOLVE-BLOCK-START
def _factorize(n):
    """Return prime factorization of n as dict {prime: exponent}."""
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

def _divisors_from_factors(factors):
    """Generate all divisors from prime factorization."""
    primes = list(factors.keys())
    exps = [factors[p] for p in primes]
    divisors = [1]
    for i, p in enumerate(primes):
        current = []
        for d in divisors:
            mult = 1
            for e in range(exps[i] + 1):
                current.append(d * mult)
                mult *= p
        divisors = current
    return divisors

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s wall-clock
    
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x must be such that 4/x < 4/n => x > n/4
        # and 3/x > 4/n => x < 3n/4
        x_start = n // 4 + 1
        x_end = (3 * n) // 4
        
        for x in range(x_start, x_end + 1):
            if time.time() > deadline:
                break
            
            K = 4 * x - n
            if K <= 0:
                continue
            
            # We need A * B = (n*x)^2
            # such that (A + n*x) % K == 0 and (B + n*x) % K == 0
            target = n * x
            target_sq = target * target
            
            # Factorize target_sq and generate divisors
            fac = _factorize(target_sq)
            divisors = _divisors_from_factors(fac)
            
            for A in divisors:
                B = target_sq // A
                if (A + target) % K == 0:
                    if (B + target) % K == 0:
                        y = (A + target) // K
                        z = (B + target) // K
                        # Ensure distinctness: x, y, z all distinct and positive
                        if x != y and x != z and y != z:
                            if x > 0 and y > 0 and z > 0:
                                if z <= 10**12 and y <= 10**12:
                                    found = [x, y, z]
                                    break
            if found:
                break
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
