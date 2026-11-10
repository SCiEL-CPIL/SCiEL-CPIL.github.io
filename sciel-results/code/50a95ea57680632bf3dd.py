import time
import random

def construct():
    t0 = time.time()
    deadline = t0 + 75.0

    def nz_square(coeffs):
        """Count nonzero entries in the square of polynomial given as list."""
        nz = [(i, c) for i, c in enumerate(coeffs) if c]
        sq = {}
        for i, ci in nz:
            for j, cj in nz:
                d = i + j
                sq[d] = sq.get(d, 0) + ci * cj
        return sum(1 for v in sq.values() if v)

    def make_poly(pairs):
        if not pairs:
            return [0]
        md = max(d for d, _ in pairs)
        p = [0] * (md + 1)
        for d, c in pairs:
            p[d] = c
        return p

    def valid(p):
        return (len(p) <= 41 and p[-1] and
                sum(1 for x in p if x) == 12 and
                all(-64 <= x <= 64 for x in p))

    best = 23
    best_p = [1] * 12

    rng = random.Random(42)

    # Phase 1: random search
    for _ in range(200000):
        if time.time() > deadline:
            break
        md = rng.randint(8, 40)
        degs = rng.sample(range(md + 1), 12)
        degs.sort()
        cs = [rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]) for _ in range(12)]
        p = make_poly(zip(degs, cs))
        if not valid(p):
            continue
        n = nz_square(p)
        if n < best:
            best = n
            best_p = p
            if best <= 13:
                break

    return best_p
