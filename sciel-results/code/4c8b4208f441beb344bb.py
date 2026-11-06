import time
import math

def _factorize(num):
    """Return prime factorization as dict {prime: exponent}."""
    factors = {}
    d = 2
    while d * d <= num:
        while num % d == 0:
            factors[d] = factors.get(d, 0) + 1
            num //= d
        d += 1
    if num > 1:
        factors[num] = factors.get(num, 0) + 1
    return factors

def _divisors_from_factors(factors):
    """Generate all divisors from prime factorization dict."""
    primes = list(factors.keys())
    exponents = [factors[p] for p in primes]
    divisors = [1]
    for p, e in zip(primes, exponents):
        current = []
        power = 1
        for i in range(e + 1):
            for d in divisors:
                current.append(d * power)
            power *= p
        divisors = current
    return divisors

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    results = [None] * 254
    MAX_DENOM = 10**12
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        n_sq = n * n
        x_start = n // 4 + 1
        
        # Fast path: look for x such that 4x-n divides n^2
        # This implies A_red = 1, leading to a simple formula for y, z
        for x in range(x_start, x_start + 100):
            if time.time() > deadline:
                break
            d = 4 * x - n
            if d <= 0:
                continue
            if n_sq % d == 0:
                # A_red = 1
                B = n * x
                B_red = B // d
                y = B_red + 1
                z = B_red * (B_red + 1)
                if y > x and z > y and y <= MAX_DENOM and z <= MAX_DENOM:
                    found = [x, y, z]
                    break
        
        # General path: if fast path failed, use divisor enumeration for B_red^2
        if not found:
            for x in range(x_start, x_start + 20):
                if time.time() > deadline:
                    break
                A = 4 * x - n
                if A <= 0:
                    continue
                B = n * x
                g = math.gcd(A, B)
                A_red = A // g
                B_red = B // g
                
                # Factorize B_red
                fact_B_red = _factorize(B_red)
                # Factorize B_red^2
                fact_B_red_sq = {}
                for p, e in fact_B_red.items():
                    fact_B_red_sq[p] = 2 * e
                
                divs = _divisors_from_factors(fact_B_red_sq)
                
                for u in divs:
                    if u >= B_red:
                        continue
                    # Check u = -B_red (mod A_red)
                    if (u + B_red) % A_red != 0:
                        continue
                    
                    y = (B_red + u) // A_red
                    if y <= x:
                        continue
                    v = (B_red * B_red) // u
                    z = (B_red + v) // A_red
                    if z <= y:
                        continue
                    if y > MAX_DENOM or z > MAX_DENOM:
                        continue
                    found = [x, y, z]
                    break
                if found:
                    break
        
        results[idx] = found
    
    return results
