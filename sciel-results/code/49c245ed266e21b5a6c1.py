import math
import time

def _factorize(n):
    """Return prime factorization of n as a list of (prime, exponent)."""
    if n < 2:
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
    return factors

def _gen_divisors(factors):
    """Generate all divisors from a list of (prime, exponent) pairs."""
    divisors = [1]
    for p, e in factors:
        current_divisors = []
        for d in divisors:
            p_power = 1
            for _ in range(e + 1):
                current_divisors.append(d * p_power)
                p_power *= p
        divisors = current_divisors
    return divisors

def _find_solution(n):
    """Find distinct positive integers x,y,z with 1/x+1/y+1/z=4/n, x<y<z, z<=10^12."""
    # 1/x + 1/y + 1/z = 4/n
    # Fix x. Then 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx)
    # Let A = 4x - n, B = nx.
    # 1/y + 1/z = A/B
    # (Ay - B)(Az - B) = B^2
    # Let u = Ay - B, v = Az - B. Then uv = B^2.
    # y = (u + B) / A, z = (v + B) / A.
    # Conditions:
    # 1. A > 0 => 4x > n => x > n/4.
    # 2. y, z integers => A | (u+B) and A | (v+B).
    #    Note: u v = B^2. Also u ≡ -B (mod A) and v ≡ -B (mod A).
    #    Since uv = B^2, if u ≡ -B (mod A), then v = B^2/u.
    #    We need to check u such that u | B^2 and u ≡ -B (mod A).
    # 3. y < z => u < v => u < B (since u+v is fixed? No, u v = B^2. If u < B, then u < v).
    #    Actually, if u < B, then v = B^2/u > B. So u < B implies u < v.
    # 4. x < y => x < (u+B)/A => Ax < u + B => u > Ax - B.
    #    So u > x(4x - n) - nx = 4x^2 - 2nx.
    # 5. z <= 10^12.
    # 6. Distinct: x != y, y != z, x != z.
    #    y != z is guaranteed by u != v (u < B).
    #    x != y: if x == y, then 1/x + 1/x + 1/z = 4/n.
    #    x != z: if x == z, then 1/x + 1/y + 1/x = 4/n.
    
    # Iterate x
    # x must be > n/4.
    # Upper bound for x?
    # 1/x < 4/n => x > n/4.
    # 1/x + 1/(x+1) + 1/z = 4/n.
    # If x is very large, 1/x is small.
    # Max x is when y=x+1, z is large?
    # Actually, we just need *one* solution.
    # Usually small x works.
    
    # To be safe, iterate x up to some limit.
    # For n=3, x > 0.75. x=1, 2, 3...
    # For n=256, x > 64.
    
    # Let's estimate the max x needed.
    # If we pick x, we need a divisor u of B^2 such that u > 4x^2 - 2nx and u < B and u ≡ -B (mod A).
    
    # Optimization: Factorize n once.
    # For each x, factorize B = nx.
    # Generate divisors of B^2.
    # Filter by u < B, u > min_u, u ≡ -B (mod A).
    
    n_factors = _factorize(n)
    
    # We can try x in increasing order.
    # How far?
    # Let's try up to n + 1000 or something reasonable.
    # Actually, for 4/n, solutions are dense.
    
    for x in range(n // 4 + 1, n // 4 + 2000):
        A = 4 * x - n
        if A <= 0:
            continue
        B = n * x
        B2 = B * B
        
        min_u = x * A - B + 1
        if min_u < 1:
            min_u = 1
            
        # If min_u >= B, no solution for this x (since u < B for y < z)
        if min_u >= B:
            continue
            
        # Factorize B
        # B = n * x
        # We have n_factors. We need x_factors.
        x_factors = _factorize(x)
        
        # Combine factors for B
        # Use a dict to sum exponents
        b_factors_dict = {}
        for p, e in n_factors:
            b_factors_dict[p] = e
        for p, e in x_factors:
            b_factors_dict[p] = b_factors_dict.get(p, 0) + e
            
        # Factors for B^2 are (p, 2*e)
        b2_factors = [(p, 2*e) for p, e in b_factors_dict.items()]
        
        # Generate divisors of B^2
        divs = _gen_divisors(b2_factors)
        
        # Find u in divs such that:
        # 1. min_u <= u < B
        # 2. (u + B) % A == 0
        
        for u in divs:
            if u < min_u:
                continue
            if u >= B:
                continue # We want u < v, so u < B. If u=B, v=B, y=z, not distinct.
            
            if (u + B) % A != 0:
                continue
            
            v = B2 // u
            # Check v condition (should be symmetric if u works, but let's be sure)
            # v = B^2 / u.
            # We need (v + B) % A == 0.
            # Note: u v = B^2.
            # If u ≡ -B (mod A), then u = kA - B.
            # This doesn't immediately imply v ≡ -B (mod A).
            # Let's check explicitly.
            if (v + B) % A != 0:
                continue
                
            y = (u + B) // A
            z = (v + B) // A
            
            # Check distinctness and bounds
            if y <= x or z <= y:
                continue
            if z > 10**12:
                continue
            if x == y or y == z or x == z:
                continue
                
            return [x, y, z]
            
    return None

def construct():
    deadline = time.time() + 75  # 75s safety margin (90s limit)
    answers = [None] * 254  # for n=3..256
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = _find_solution(n)
        answers[idx] = found
        
    return answers
