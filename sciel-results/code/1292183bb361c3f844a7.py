import time
from functools import lru_cache

@lru_cache(maxsize=None)
def factorize(n):
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 1)
    return tuple(sorted(factors.items()))

@lru_cache(maxsize=None)
def get_divisors_of_sq(n):
    """Get sorted divisors of n^2, cached."""
    fac = factorize(n)
    fac_sq = tuple((p, 2 * e) for p, e in fac)
    divs = [1]
    for p, e in fac_sq:
        new_divs = []
        for dv in divs:
            val = dv
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divs = new_divs
    divs.sort()
    return tuple(divs)

def construct():
    deadline = time.time() + 82.0
    answers = [None] * 254
    BOUND = 10**12

    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break

        found = None
        x_min = n // 4 + 1

        # Phase 1: Divisor-based search with moderate x range
        x_end_phase1 = max(4 * n + 50, 2000)
        for x in range(x_min, x_end_phase1):
            if time.time() > deadline:
                break

            A = 4 * x - n
            if A <= 0:
                continue

            B = n * x
            B_sq = B * B

            divisors = get_divisors_of_sq(B)

            for u in divisors:
                if u > B:
                    break
                if (u + B) % A != 0:
                    continue

                v = B_sq // u
                if (v + B) % A != 0:
                    continue

                y = (u + B) // A
                z = (v + B) // A

                if y <= 0 or z <= 0:
                    continue
                if y > BOUND or z > BOUND:
                    continue
                if x == y or x == z or y == z:
                    continue

                # Verify by cross multiplication
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break

            if found:
                break

        # Phase 2: Extended x range with divisor search
        if not found and time.time() < deadline:
            x_start_phase2 = x_end_phase1
            x_end_phase2 = 50000
            for x in range(x_start_phase2, x_end_phase2):
                if time.time() > deadline:
                    break

                A = 4 * x - n
                if A <= 0:
                    continue

                B = n * x
                B_sq = B * B

                divisors = get_divisors_of_sq(B)

                for u in divisors:
                    if u > B:
                        break
                    if (u + B) % A != 0:
                        continue

                    v = B_sq // u
                    if (v + B) % A != 0:
                        continue

                    y = (u + B) // A
                    z = (v + B) // A

                    if y <= 0 or z <= 0:
                        continue
                    if y > BOUND or z > BOUND:
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

        # Phase 3: Direct y-scan fallback for remaining unsolved n
        if not found and time.time() < deadline:
            # For 1/x + 1/y + 1/z = 4/n, fix x and scan y
            # 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx) = A/B
            # So y*z / (y+z) = B/A, i.e., (y - B/A)(z - B/A) = B^2/A^2
            # Try small x values and scan y in a reasonable range
            for x in range(x_min, min(x_min + 500, 5000)):
                if time.time() > deadline:
                    break

                A = 4 * x - n
                if A <= 0:
                    continue

                B = n * x

                # y must satisfy: A*y*z = B*(y+z) => A*y*z - B*y - B*z = 0
                # => (A*y - B)(A*z - B) = B^2
                # Let p = A*y - B, q = A*z - B, then p*q = B^2
                # y = (p + B)/A, z = (q + B)/A
                # We already did this via divisors. Instead, try scanning y directly.
                # From 1/y + 1/z = A/B: z = B*y / (A*y - B)
                # Need A*y - B > 0, i.e., y > B/A
                y_min = B // A + 1
                y_max = min(y_min + 2000, 10**6)

                for y in range(y_min, y_max):
                    if time.time() > deadline:
                        break

                    denom = A * y - B
                    if denom <= 0:
                        continue
                    if (B * y) % denom != 0:
                        continue

                    z = (B * y) // denom
                    if z <= 0 or z > BOUND:
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
