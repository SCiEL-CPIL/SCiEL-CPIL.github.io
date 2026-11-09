"""Direct construction of distinct three-term unit-fraction representations of 4/n."""
# EVOLVE-BLOCK-START
import time
import math

def _factorize(num):
    """Factorize a number into prime factors."""
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
    """Generate all divisors from prime factorization."""
    divisors = [1]
    for p, e in factors.items():
        new_divs = []
        for d in divisors:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
    return divisors

def _divisors_of_b2(b):
    """Get all divisors of b^2."""
    factors = _factorize(b)
    # For b^2, each exponent doubles
    factors_b2 = {p: 2 * e for p, e in factors.items()}
    return _divisors_from_factors(factors_b2)

def _try_x(n, x, BOUND):
    """Try a specific x value; return [x,y,z] if a valid distinct triple found, else None."""
    a = 4 * x - n
    b = n * x
    if a <= 0 or b <= 0:
        return None
    b2 = b * b
    divs = _divisors_of_b2(b)
    for d in divs:
        if d <= 0 or d >= b:
            continue
        num_y = d + b
        if num_y % a != 0:
            continue
        y = num_y // a
        if y <= x:
            continue
        d2 = b2 // d
        num_z = d2 + b
        if num_z % a != 0:
            continue
        z = num_z // a
        if z <= y:
            continue
        if x == y or y == z or x == z:
            continue
        if z > BOUND or y > BOUND or x > BOUND:
            continue
        # Verify via cross multiplication
        lhs_num = n * y * z + n * x * z + n * x * y
        rhs_num = 4 * x * y * z
        if lhs_num == rhs_num:
            return [x, y, z]
    return None

def construct():
    deadline = time.time() + 85  # safety margin before the 90s kill
    answers = []
    BOUND = 10**12

    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue

        found = None

        # Priority 1: try x = n+1 first (the classic identity, works for all n).
        # This is the fastest success path and resolves nearly every n instantly.
        found = _try_x(n, n + 1, BOUND)

        # Priority 2: if that failed, sweep x from x_min upward (small x = small a = cheap divisor search).
        if not found:
            x_min = n // 4 + 1
            x_max = (3 * n) // 4
            # Small x first: a = 4x - n is small, so b/a ratio is large but divisor count of b^2 is manageable;
            # more importantly, small x tends to yield solutions quickly.
            for x in range(x_min, x_max + 1):
                if time.time() > deadline:
                    break
                if found:
                    break
                if x == n + 1:
                    continue  # already tried
                found = _try_x(n, x, BOUND)

        # Priority 3: if still not found, sweep x from x_max downward as a final fallback.
        if not found:
            x_min = n // 4 + 1
            x_max = (3 * n) // 4
            for x in range(x_max, x_min - 1, -1):
                if time.time() > deadline:
                    break
                if found:
                    break
                if x == n + 1:
                    continue
                found = _try_x(n, x, BOUND)

        answers.append(found)
    return answers
# EVOLVE-BLOCK-END
