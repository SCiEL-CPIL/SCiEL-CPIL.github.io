import time
import math

DENOM_BOUND = 10**12

def _factorize(num):
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
    divs = [1]
    for p, e in factors.items():
        new_divs = []
        pe = 1
        for i in range(e + 1):
            for d in divs:
                new_divs.append(d * pe)
            pe *= p
        divs = new_divs
    return divs

def _try_x(n, x, deadline, t_start, budget):
    """Try a specific x value for given n. Returns [x,y,z] or None."""
    if time.time() > deadline:
        return None
    A = 4 * x - n
    if A <= 0:
        return None
    B = n * x
    u_min = A * x - B + 1
    u_max = B - 1
    if u_min > u_max:
        return None

    B_sq = B * B
    B_factors = _factorize(B)
    B_sq_factors = {p: 2 * e for p, e in B_factors.items()}
    B_sq_divs = _divisors_from_factors(B_sq_factors)
    
    # Sort divisors to try smaller ones first (more likely to give valid y,z)
    B_sq_divs.sort()
    
    for u in B_sq_divs:
        if u < u_min:
            continue
        if u > u_max:
            break
        if B_sq % u != 0:
            continue
        v = B_sq // u
        if v <= u:
            continue
        if (u + B) % A != 0:
            continue
        if (v + B) % A != 0:
            continue
        y = (u + B) // A
        z = (v + B) // A
        if y <= x or z <= y:
            continue
        if z > DENOM_BOUND:
            continue
        if x == y or y == z or x == z:
            continue
        return [x, y, z]
    return None

def construct():
    deadline = time.time() + 85.0
    results = [None] * 254

    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break

        found = None
        t_start = time.time()
        remaining = deadline - t_start
        # Allocate more time to harder (larger n) cases
        n_budget = min(0.3, remaining / max(1, 256 - idx))

        x_min = n // 4 + 1
        x_max = 3 * n // 4

        # Try promising x values first: x=n, then x near n, then expand
        # x=n gives A=3n which is a nice structure
        candidate_xs = []
        # Add x=n if in range
        if x_min <= n <= x_max:
            candidate_xs.append(n)
        # Add x values near n
        for dx in range(1, 50):
            for x in [n - dx, n + dx]:
                if x_min <= x <= x_max and x not in candidate_xs:
                    candidate_xs.append(x)
        # Then add remaining in order
        for x in range(x_min, x_max + 1):
            if x not in candidate_xs:
                candidate_xs.append(x)
        
        for x in candidate_xs:
            if time.time() > deadline or (time.time() - t_start) > n_budget:
                break
            found = _try_x(n, x, deadline, t_start, n_budget)
            if found:
                break

        results[idx] = found

    return results
