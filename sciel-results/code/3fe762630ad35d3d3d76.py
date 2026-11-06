def construct():
    import time, random
    from math import lcm

    t0 = time.time()
    DEADLINE = 47.0
    N = 128
    elems = list(range(1, N + 1))

    # ---- LCM matrix (symmetric) ----
    lcm_m = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        row = lcm_m[i]
        for j in range(i, N + 1):
            v = lcm(i, j)
            row[j] = v
            lcm_m[j][i] = v

    # ---- Bad triples: for each vertex, the pairs that with it form a bad triple ----
    bad_pairs = {x: [] for x in elems}
    for a in range(1, N + 1):
        row_a = lcm_m[a]
        for b in range(a + 1, N + 1):
            ab = row_a[b]
            row_b = lcm_m[b]
            for c in range(b + 1, N + 1):
                if ab == row_a[c] and ab == row_b[c]:
                    bad_pairs[a].append((b, c))
                    bad_pairs[b].append((a, c))
                    bad_pairs[c].append((a, b))

    degree = {x: len(bad_pairs[x]) for x in elems}

    def can_add(x, S):
        for (y, z) in bad_pairs[x]:
            if y in S and z in S:
                return False
        return True

    def greedy(order):
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S

    # ---- Strong deterministic starts ----
    orders = [
        elems,
        list(reversed(elems)),
        sorted(elems, key=lambda x: degree[x]),
        sorted(elems, key=lambda x: -degree[x]),
    ]
    best = None
    best_size = 0
    for o in orders:
        S = greedy(o)
        if len(S) > best_size:
            best = S
            best_size = len(S)

    rng = random.Random(12345)
    for _ in range(300):
        o = elems[:]
        rng.shuffle(o)
        S = greedy(o)
        if len(S) > best_size:
            best = S
            best_size = len(S)

    def time_left():
        return DEADLINE - (time.time() - t0)

    deg_asc = sorted(elems, key=lambda x: degree[x])
    k_pool = [1, 1, 2, 2, 3, 3, 4, 5, 7, 9]

    # ---- Iterated local search: kick from best, regrow with varied ordering ----
    while time_left() > 1.5:
        k = rng.choice(k_pool)
        S2 = set(best)
        lst = list(S2)
        rng.shuffle(lst)
        for x in lst[:min(k, len(S2))]:
            S2.discard(x)
        r = rng.random()
        if r < 0.30:
            order = elems
        elif r < 0.60:
            order = deg_asc
        elif r < 0.85:
            order = elems[:]
            rng.shuffle(order)
        else:
            # full random restart
            S2 = set()
            order = elems[:]
            rng.shuffle(order)
        for x in order:
            if x not in S2 and can_add(x, S2):
                S2.add(x)
        if len(S2) > best_size:
            best = S2
            best_size = len(S2)

    # ---- Final polish: single removal + regrow with a few orderings ----
    if time_left() > 0.5:
        for e in list(best):
            if time_left() < 0.5:
                break
            for order in (deg_asc, elems):
                S2 = set(best)
                S2.discard(e)
                for x in order:
                    if x not in S2 and can_add(x, S2):
                        S2.add(x)
                if len(S2) > best_size:
                    best = S2
                    best_size = len(S2)

    return list(best)
