def construct():
    from fractions import Fraction
    import time

    target = Fraction(5, 31)
    start_time = time.time()
    deadline = start_time + 55

    best = None
    best_max = float('inf')

    def dfs(rem, min_d, path, depth):
        nonlocal best, best_max
        if time.time() > deadline:
            return
        if rem == 0:
            if path:
                m = max(path)
                if m < best_max:
                    best_max = m
                    best = list(path)
            return
        if rem < 0 or depth <= 0:
            return
        d = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
        d_end = d + 300
        for d in range(d, d_end + 1):
            if time.time() > deadline:
                return
            t = Fraction(1, d)
            if t > rem:
                break
            nr = rem - t
            path.append(d)
            dfs(nr, d + 1, path, depth - 1)
            path.pop()

    for max_depth in range(3, 8):
        if time.time() > deadline:
            break
        for first in range(7, 25):
            if time.time() > deadline:
                break
            rem = target - Fraction(1, first)
            if rem < 0:
                continue
            dfs(rem, first + 1, [first], max_depth - 1)

    if best is not None:
        return best

    r = target
    values = []
    while r > 0:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
