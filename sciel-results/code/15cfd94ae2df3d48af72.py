import time

def construct():
    t0 = time.time()
    deadline = t0 + 40.0
    n = 12
    MAXV = 1048576

    base = [1 << i for i in range(n)]

    def max_of(a):
        m = 0
        for x in a:
            if x > m:
                m = x
        return m

    # Greedy minimal superincreasing (each new > sum of prior => distinct subset sums)
    def greedy(start):
        out = [start]
        s = start
        for _ in range(n - 1):
            s += 1
            out.append(s)
        if out[-1] > MAXV:
            return None
        return out

    def valid(a):
        if len(a) != n:
            return False
        if len(set(a)) != n:
            return False
        for x in a:
            if x <= 0 or x > MAXV:
                return False
        sums = {0}
        for x in a:
            for s in list(sums):
                sums.add(s + x)
        return len(sums) == (1 << n)

    best = base
    best_max = max_of(base)

    # Deterministic greedy starts to populate the candidate pool
    for s in (2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377):
        if time.time() > deadline:
            break
        g = greedy(s)
        if g is None:
            continue
        if valid(g):
            m = max_of(g)
            if best_max < m <= MAXV:
                best = g
                best_max = m

    # Randomized refinement: start from a greedy seed, randomly bump one element
    # (breaking superincreasing) then try to repair distinctness by raising elements.
    import random
    rng = random.Random(12345)
    it = 0
    while time.time() < deadline:
        it += 1
        g = greedy(rng.randint(1, 40))
        if g is None or not valid(g):
            continue
        cur = list(g)
        # randomly try to lower the max by reducing a large element (if still valid)
        idx = rng.randrange(2, n)
        lo = 1
        hi = cur[idx] - 1
        if hi > lo:
            for _ in range(30):
                if time.time() > deadline:
                    break
                cand = list(cur)
                cand[idx] = rng.randint(lo, hi)
                if cand[idx] != cur[idx] and valid(cand):
                    cur = cand
        m = max_of(cur)
        if best_max < m <= MAXV:
            best = cur
            best_max = m

    return best
