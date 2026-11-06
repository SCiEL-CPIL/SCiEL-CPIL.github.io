import time
import math

def _factorize(num):
    """Return prime factorization of num as dict {prime: exponent}."""
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

def _divisors_from_factorization(factors):
    """Generate all divisors from a prime factorization dict."""
    divisors = [1]
    for prime, exp in factors.items():
        new_divisors = []
        power = 1
        for e in range(exp + 1):
            for d in divisors:
                new_divisors.append(d * power)
            power *= prime
        divisors = new_divisors
    return divisors

def construct():
    deadline = time.time() + 75.0
    answers = [None] * 254
    BOUND = 10**12
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        found = None
        
        x_start = n // 4 + 1
        # Broader x range: go up to about 3*n to give more chances
        x_end = min(3 * n + 50, 2000)
        
        # Phase 1: Direct y-scan for each x (fast, bounded y range)
        # For each x, try y in a reasonable range and compute z directly
        max_y_scan = 50000
        for x in range(x_start, x_end + 1):
            if time.time() > deadline:
                break
            if found:
                break
            
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # y must satisfy: A*y - B > 0 => y > B/A
            # And z = B*y/(A*y - B) > y (for distinctness with y < z)
            # z > y => B*y > y*(A*y - B) => B > A*y - B => 2B > A*y => y < 2B/A
            # So y in (B/A, 2B/A)
            y_min = B // A + 1
            y_max = min(2 * B // A, y_min + max_y_scan)
            
            for y in range(y_min, y_max + 1):
                if time.time() > deadline:
                    break
                denom = A * y - B
                if denom <= 0:
                    continue
                if (B * y) % denom == 0:
                    z = (B * y) // denom
                    if z > y and z != x and z <= BOUND and y <= BOUND and x <= BOUND:
                        found = [x, y, z]
                        break
            if found:
                break
        
        # Phase 2: If not found, try divisor enumeration of B^2 for selected x values
        if not found:
            # Try a subset of x values with divisor enumeration
            x_candidates = list(range(x_start, min(x_start + 20, x_end + 1)))
            if x_end - x_start > 20:
                step = max(1, (x_end - x_start) // 5)
                for i in range(1, 6):
                    x_val = x_start + i * step
                    if x_val <= x_end and x_val not in x_candidates:
                        x_candidates.append(x_val)
            
            for x in x_candidates:
                if time.time() > deadline:
                    break
                if found:
                    break
                
                A = 4 * x - n
                if A <= 0:
                    continue
                B = n * x
                B_sq = B * B
                
                factors_B = _factorize(B)
                factors_B_sq = {p: 2 * e for p, e in factors_B.items()}
                divs = _divisors_from_factorization(factors_B_sq)
                
                for u in divs:
                    v = B_sq // u
                    if (u + B) % A != 0:
                        continue
                    if (v + B) % A != 0:
                        continue
                    
                    y = (u + B) // A
                    z = (v + B) // A
                    
                    if y <= 0 or z <= 0:
                        continue
                    if y > BOUND or z > BOUND or x > BOUND:
                        continue
                    if x == y or x == z or y == z:
                        continue
                    
                    lhs = n * (y * z + x * z + x * y)
                    rhs = 4 * x * y * z
                    if lhs == rhs:
                        found = [x, y, z]
                        break
                if found:
                    break
        
        answers[idx] = found
    
    return answers
