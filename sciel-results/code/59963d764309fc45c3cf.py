def construct():
    import time
    from fractions import Fraction
    start = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    target = Fraction(5, 31)
    SEED = [8, 31, 248]
    best_val = 248
    best_sol = SEED

    def two_term(r, lo, M, used):
        # find (b,c), b<c, b>=lo, c<=M, distinct from used, 1/b+1/c=r
        p, q = r.numerator, r.denominator
        # (p b - q)(p c - q) = q^2
        q2 = q * q
        res = []
        # iterate divisors X of q2 with X <= q
        i = 1
        # get all divisors
        divs = []
        while i * i <= q2:
            if q2 % i == 0:
                divs.append(i)
                if i != q2 // i:
                    divs.append(q2 // i)
            i += 1
        for X in divs:
            Y = q2 // X
            if X > Y:
                continue
            if X == Y:
                continue  # b==c not allowed
            if (X + q) % p != 0:
                continue
            if (Y + q) % p != 0:
                continue
            b = (X + q) // p
            c = (Y + q) // p
            if b <= 0 or c <= 0:
                continue
            if b >= c:
                continue
            if b < lo:
                continue
            if c > M:
                continue
            if b in used or c in used:
                continue
            res.append((b, c))
        return res

    def dfs(r, last, used, chosen, M):
        # returns list of denominators (full solution) or None
        if r == 0:
            return chosen[:]
        # 1-term completion
        if r.numerator == 1:
            d = r.denominator
            if d > last and d <= M and d not in used:
                return chosen + [d]
        # 2-term completion
        if time.time() > DEADLINE:
            return None
        for (b, c) in two_term(r, last + 1, M, used):
            return chosen + [b, c]
        # recurse
        # next d: 1/d < r  => d > 1/r ; and d >= last+1 ; d <= M
        import math
        lo = max(last + 1, (r.denominator + r.numerator - 1) // r.numerator)  # ceil(1/r)
        for d in range(lo, M + 1):
            nr = r - Fraction(1, d)
            if nr <= 0:
                continue
            # pruning: nr must be representable by terms > d; the max single term is 1/(d+1)
            # if nr > sum of available, skip (approx: nr must be < 1/(d+1) + ... ) keep light
            used2 = used | {d}
            sol = dfs(nr, d, used2, chosen + [d], M)
            if sol is not None:
                return sol
            if time.time() > DEADLINE:
                return None
        return None

    for M in range(8, best_val):
        if time.time() > DEADLINE:
            break
        sol = dfs(target, 0, set(), [], M)
        if sol is not None:
            # verify
            s = sum(Fraction(1, x) for x in sol)
            if s == target and len(set(sol)) == len(sol) and all(1 <= x <= 10000000 for x in sol):
                if max(sol) <= M:
                    best_val = max(sol)
                    best_sol = sol
                    break
    return best_sol
