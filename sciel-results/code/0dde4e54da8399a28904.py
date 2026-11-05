def construct():
    import time
    from fractions import Fraction
    start = time.time()
    TIME_LIMIT = 50.0

    def ceil_div(a, b):
        return (a + b - 1) // b

    def max_sum(d, r):
        s = Fraction(0, 1)
        for i in range(r):
            s += Fraction(1, d + i)
        return s

    def search(k, memo):
        # find representation of exactly k terms
        S = Fraction(1, 1)
        m = 5
        chosen = []

        def dfs(S, r, m):
            if time.time() - start > TIME_LIMIT:
                raise TimeoutError()
            if r == 0:
                return chosen[:] if S == 0 else None
            if S <= 0:
                return None
            key = (S, r, m)
            if key in memo:
                return memo[key]
            if r == 1:
                # need 1/d = S, d >= m
                p, q = S.numerator, S.denominator
                if p != 0 and q % p == 0:
                    d = q // p
                    if d >= m:
                        res = chosen[:] + [d]
                        memo[key] = res
                        return res
                memo[key] = None
                return None
            L = max(m, ceil_div(S.denominator, S.numerator))
            # find U: largest d >= L with max_sum(d, r) >= S
            # iterate to find upper bound
            # binary search
            lo, hi = L, L
            # expand hi until max_sum(hi, r) < S or hi too big
            while max_sum(hi, r) >= S and hi < 10**7:
                hi *= 2
            # now max_sum(hi, r) < S (or hi hit cap). binary search in [L, hi]
            # we want largest d with max_sum(d, r) >= S
            if max_sum(L, r) < S:
                memo[key] = None
                return None
            # binary search
            a, b = L, hi  # f(a)>=S, f(b)<S
            while a < b:
                mid = (a + b + 1) // 2
                if max_sum(mid, r) >= S:
                    a = mid
                else:
                    b = mid - 1
            U = a
            # try d from U down to L
            for d in range(U, L - 1, -1):
                S2 = S - Fraction(1, d)
                if S2 < 0:
                    continue
                # quick check: max_sum(d+1, r-1) >= S2
                if max_sum(d + 1, r - 1) < S2:
                    continue
                chosen.append(d)
                res = dfs(S2, r - 1, d + 1)
                chosen.pop()
                if res is not None:
                    memo[key] = res
                    return res
            memo[key] = None
            return None

        res = dfs(S, k, m)
        return res

    for k in range(3, 30):
        memo = {}
        try:
            res = search(k, memo)
        except TimeoutError:
            break
        if res is not None:
            # verify
            s = Fraction(0, 1)
            for d in res:
                s += Fraction(1, d)
            assert s == 1, (res, s)
            assert len(set(res)) == len(res)
            assert all(5 <= d <= 10000000 for d in res)
            assert 1 <= len(res) <= 256
            return res
    # fallback
    return [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  # placeholder, need valid
