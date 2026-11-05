def construct():
    from fractions import Fraction
    import time

    target = Fraction(5, 31)
    start_time = time.time()
    TIMEOUT = 50

    def greedy(r):
        vals = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            vals.append(d)
            r -= Fraction(1, d)
        return vals

    greedy_sol = greedy(target)
    best_max = [max(greedy_sol)]
    best_sol = list(greedy_sol)

    def search(remaining, min_den, current):
        nonlocal best_sol
        if time.time() - start_time > TIMEOUT:
            return
        if not remaining:
            if current:
                m = max(current)
                if m < best_max[0]:
                    best_max[0] = m
                    best_sol = list(current)
            return
        if len(current) >= 256:
            return
        low = max(min_den, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        if low >= best_max[0]:
            return
        k = 256 - len(current)
        if Fraction(k, low) < remaining:
            return
        for d in range(low, best_max[0]):
            if time.time() - start_time > TIMEOUT:
                return
            term = Fraction(1, d)
            if term > remaining:
                continue
            new_rem = remaining - term
            current.append(d)
            search(new_rem, d + 1, current)
            current.pop()

    search(target, 1, [])
    return best_sol
