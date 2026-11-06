import time

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

def construct():
    deadline = time.time() + 85.0
    results = [None] * 254

    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break

        found = None
        t_start = time.time()
        remaining = deadline - t_start
        n_budget = min(0.25, remaining / max(1, 256 - idx))

        x_min = n // 4 + 1
        x_max = 3 * n // 4

        for x in range(x_min, x_max + 1):
            if time.time() > deadline or (time.time() - t_start) > n_budget:
                break

            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            u_min = A * x - B + 1
            u_max = B - 1
            if u_min > u_max:
                continue

            B_sq = B * B

            B_factors = _factorize(B)
            B_sq_factors = {p: 2 * e for p, e in B_factors.items()}
            B_sq_divs = _divisors_from_factors(B_sq_factors)

            for u in B_sq_divs:
                if u < u_min or u > u_max:
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
                found = [x, y, z]
                break

            if found:
                break

        results[idx] = found

    return results
