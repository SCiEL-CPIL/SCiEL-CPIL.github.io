import math

def _sf(n):
    """Compute square-free part of n."""
    sf = 1
    p = 2
    x = n
    while p * p <= x:
        c = 0
        while x % p == 0:
            x //= p
            c += 1
        if c % 2 == 1:
            sf *= p
        p += 1 if p == 2 else 2
    if x > 1:
        sf *= x
    return sf

def construct():
    # 1. Single element solution
    # s * 210 = k^2 => s = 210 * m^2
    # m=1 => 210 (not > 210)
    # m=2 => 840
    best_sol = [840]
    best_max = 840

    # 2. Pair solution
    # Iterate a from 211 to best_max - 1
    # For each a, find smallest b > a such that sf(a*b) = 210
    # sf(a*b) = 210 => sf(b) = sf(a*210)
    for a in range(211, best_max):
        v = _sf(a * 210)
        # Find smallest b > a such that sf(b) = v
        # b = v * k^2
        # v * k^2 > a => k > sqrt(a/v)
        k = math.isqrt(a // v) + 1
        b = v * k * k
        if b > a and b <= 10210:
            if b < best_max:
                best_max = b
                best_sol = [a, b]
        # If a >= best_max, we can break, but loop condition handles it
        # Actually, if we update best_max, the range is fixed at start of loop in Python?
        # No, range(211, best_max) is evaluated once.
        # So we might iterate a up to 839 even if we find b=315.
        # That's fine, 839-211 is small.
        # But we can break early if a >= current best_max found in this loop?
        # Let's just let it run, it's fast.

    # 3. Triplet solution with max < best_max
    # Precompute min_a for each square-free value v
    # min_a[v] = smallest a > 210 such that sf(a) = v
    min_a_for_sf = {}
    for a in range(211, 10211):
        v = _sf(a)
        if v not in min_a_for_sf:
            min_a_for_sf[v] = a
        # Since we iterate a upwards, the first one is the minimum

    # Search for triplet (a, c, b) with a < c < b and b < best_max
    # Condition: sf(a*c*b) = 210 => sf(a) = sf(b*c*210)
    for b in range(211, best_max):
        for c in range(211, b):
            v_needed = _sf(b * c * 210)
            if v_needed in min_a_for_sf:
                a = min_a_for_sf[v_needed]
                if a < c:
                    # Found valid triplet
                    return [a, c, b]
    
    return best_sol
