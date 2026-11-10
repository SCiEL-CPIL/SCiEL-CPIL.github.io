"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START

def _factorize(n):
    """Return dict of prime -> exponent for n."""
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

def _get_divisors_from_factors(factors):
    """Generate all divisors from a prime factorization dict."""
    if not factors:
        return [1]
    primes = list(factors.keys())
    divisors = [1]
    for p in primes:
        e = factors[p]
        new_divs = []
        pow_p = 1
        for i in range(e + 1):
            for d in divisors:
                new_divs.append(d * pow_p)
            pow_p *= p
        divisors = new_divs
    return divisors

def _get_divisors(n):
    """Get all divisors of n using factorization."""
    if n <= 0:
        return []
    factors = _factorize(n)
    return _get_divisors_from_factors(factors)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        
        # 1/x + 1/y + 1/z = 4/n
        # Assume x <= y <= z
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
            
            # (Ay - B)(Az - B) = B^2
            # Let d = Ay - B, e = Az - B, d*e = B^2
            # y = (d+B)/A, z = (e+B)/A
            # Need: d|B^2, d+B ≡ 0 (mod A), e+B ≡ 0 (mod A)
            # y >= x, z >= y, all distinct, z <= 10^12
            
            B_sq = B * B
            
            # Get divisors of B_sq that are >= B (since y >= x >= x_min > n/4, 
            # Ay - B = A*y - B >= A*x_min - B... let's just get all and filter)
            # Actually d = Ay - B. Since y >= x >= x_min, d >= A*x_min - B.
            # But it's easier to just iterate divisors and check.
            
            # To speed up, note d must satisfy:
            # 1. d divides B_sq
            # 2. (d + B) % A == 0
            # 3. (B_sq//d + B) % A == 0
            # 4. y = (d+B)//A >= x
            # 5. z = (B_sq//d + B)//A >= y
            # 6. x, y, z all distinct
            # 7. z <= 10^12
            
            # We need d > 0 and e > 0, so d < B_sq (since e = B_sq/d > 0)
            # Also d = Ay - B. For y >= 1, d >= A - B. 
            # But more usefully: since y >= x, d = Ay - B >= Ax - B = A(4x-n)/A - B... 
            # Let's just compute: d_min = A * x - B (when y = x)
            # But we need y > x for distinctness, so d > A*x - B
            # Actually y >= x, and we need distinctness, so y > x, meaning d > A*x - B
            
            # Let's get divisors of B_sq. This could be expensive for large B.
            # B = n*x, and n up to 256, x up to 3n/4 ~ 192, so B up to ~49152.
            # B_sq up to ~2.4e9. Factorization should be fast.
            
            divs = _get_divisors(B_sq)
            
            for d in divs:
                if time.time() > deadline:
                    break
                
                e = B_sq // d
                
                # Check (d + B) % A == 0
                if (d + B) % A != 0:
                    continue
                
                # Check (e + B) % A == 0
                if (e + B) % A != 0:
                    continue
                
                y = (d + B) // A
                z = (e + B) // A
                
                # Enforce ordering and distinctness
                if y < x or z < y:
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
# EVOLVE-BLOCK-END
