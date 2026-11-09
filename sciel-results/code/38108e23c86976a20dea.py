"""Direct construction of distinct three-term unit-fraction representations of 4/n."""
import time
import math

def _factorize(n):
    """Return prime factorization of n as list of (prime, exponent)."""
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            exp = 0
            while n % d == 0:
                n //= d
                exp += 1
            factors.append((d, exp))
        d += 1 if d == 2 else 2
    if n > 1:
        factors.append((n, 1))
    return factors

def _divisors_from_factors(factors):
    """Generate all divisors from prime factorization."""
    divs = [1]
    for p, e in factors:
        new_divs = []
        pe = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * pe)
            pe *= p
        divs = new_divs
    return divs

def _find_solution(n, deadline):
    """Find one valid distinct triple (x,y,z) for 1/x+1/y+1/z=4/n."""
    BIG = 10**12
    found = None

    # Pattern 1: n ≡ 3 mod 6
    if n % 6 == 3:
        m = n // 3
        x, y, z = m, 3*m + 1, 3*m*(3*m + 1)
        if x != y and y != z and x != z and x <= BIG and y <= BIG and z <= BIG:
            if n*y*z + n*x*z + n*x*y == 4*x*y*z:
                found = (x, y, z)

    # Pattern 2: n ≡ 4 mod 8
    if found is None and n % 8 == 4:
        m = n // 4
        x, y, z = m + 2, (m + 1)*(m + 2), m*(m + 1)
        if x != y and y != z and x != z and x <= BIG and y <= BIG and z <= BIG:
            if n*y*z + n*x*z + n*x*y == 4*x*y*z:
                found = (x, y, z)

    # Pattern 3: n ≡ 5 mod 10
    if found is None and n % 10 == 5:
        m = n // 5
        x, y, z = 2*m, 5*m, 10*m
        if x != y and y != z and x != z and x <= BIG and y <= BIG and z <= BIG:
            if n*y*z + n*x*z + n*x*y == 4*x*y*z:
                found = (x, y, z)

    # Pattern 4: n divisible by 8
    if found is None and n % 8 == 0:
        x, y, z = n // 4, n // 2, n
        if x != y and y != z and x != z and x >= 1:
            if n*y*z + n*x*z + n*x*y == 4*x*y*z:
                found = (x, y, z)

    # General search using factorization
    if found is None:
        x_min = n // 4 + 1
        # Upper bound for x: 1/x < 4/n => x > n/4. Also need room for y,z > x.
        # A safe upper bound: x < 4n (generous)
        x_max = min(4 * n + 10, 5000)
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                return None
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            b_sq = b * b
            
            # Factor b_sq and enumerate divisors
            bf = _factorize(b_sq)
            divs = _divisors_from_factors(bf)
            
            for d in divs:
                if time.time() > deadline:
                    return None
                if b_sq % d != 0:
                    continue
                e = b_sq // d
                # y = (d + b) / a, z = (e + b) / a
                num_y = d + b
                num_z = e + b
                if num_y % a != 0 or num_z % a != 0:
                    continue
                y = num_y // a
                z = num_z // a
                if y <= 0 or z <= 0:
                    continue
                if y > BIG or z > BIG or x > BIG:
                    continue
                if x == y or x == z or y == z:
                    continue
                # Verify with cross multiplication
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = (x, y, z)
                    break
            if found:
                break

    return found

def _generate_solutions():
    """Generate all valid triples for n=3..256 using optimized search."""
    BIG = 10**12
    solutions = {}
    deadline = time.time() + 80  # safety margin before 90s kill
    
    for n in range(3, 257):
        if time.time() > deadline:
            solutions[n] = None
            continue
        solutions[n] = _find_solution(n, deadline)
    
    return solutions

# Pre-compute all solutions at module load time
_SOLUTIONS = _generate_solutions()

def construct():
    """Return 254 rows for n=3..256. Each row is None or three distinct positive integer denominators."""
    return [_SOLUTIONS[n] for n in range(3, 257)]
