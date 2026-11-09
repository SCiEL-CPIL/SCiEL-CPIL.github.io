def construct():
    import time
    from fractions import Fraction
    target = Fraction(5, 31)
    start = time.time()
    FALLBACK = [7, 56, 1736]  # valid, max 1736

    def find_solution(M):
        inv = [Fraction(0)] * (M + 1)
        for d in range(1, M + 1):
            inv[d] = Fraction(1, d)
        suffix = [Fraction(0)] * (M + 2)
        for d in range(M, 6, -1):
            suffix[d] = suffix[d + 1] + inv[d]
        one_over_M = inv[M]
        path = []
        nodes = [0]
        LIMIT = 3000000

        def dfs(remaining, last):
            nodes[0] += 1
            if nodes[0] > LIMIT:
                return False
            if remaining == 0:
                return True
            if remaining < 0 or last >= M:
                return False
            if remaining > suffix[last + 1]:
                return False
            if remaining < one_over_M:
                return False
            d_min = last + 1
            d_min = max(d_min, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
            for d in range(d_min, M + 1):
                path.append(d)
                if dfs(remaining - inv[d], d):
                    return True
                path.pop()
            return False

        if dfs(target, 6):
            return list(path)
        return None

    M = 7
    while M <= 10000000:
        if time.time() - start > 50:
            return FALLBACK
        sol = find_solution(M)
        if sol is not None:
            return sol
        M += 1
    return FALLBACK
