def construct():
    import time
    from fractions import Fraction

    deadline = time.time() + 55.0
    target = Fraction(5, 31)
    MAX_TERMS = 256
    MAX_DEN = 10000000

    def exists_for_D(D):
        tail_max = [Fraction(0)] * (D + 2)
        s = Fraction(0)
        for d in range(D, 0, -1):
            s += Fraction(1, d)
            tail_max[d] = s

        failed = set()

        def dfs(start, remaining, chosen):
            if time.time() > deadline:
                return None
            if remaining == 0:
                return list(chosen)
            if remaining < 0 or len(chosen) >= MAX_TERMS:
                return None
            if tail_max[start] < remaining:
                return None
            key = (start, remaining)
            if key in failed:
                return None
            lo = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
            if lo < start:
                lo = start
            for d in range(lo, D + 1):
                rem2 = remaining - Fraction(1, d)
                if rem2 != 0 and tail_max[d + 1] < rem2:
                    continue
                chosen.append(d)
                res = dfs(d + 1, rem2, chosen)
                if res is not None:
                    return res
                chosen.pop()
            failed.add(key)
            return None

        return dfs(1, target, [])

    for D in range(2, 300):
        if time.time() > deadline:
            break
        sol = exists_for_D(D)
        if sol is not None:
            assert len(set(sol)) == len(sol)
            assert all(1 <= d <= MAX_DEN for d in sol)
            assert 1 <= len(sol) <= MAX_TERMS
            assert sum(Fraction(1, d) for d in sol) == target
            return sol

    r = target
    values = []
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
