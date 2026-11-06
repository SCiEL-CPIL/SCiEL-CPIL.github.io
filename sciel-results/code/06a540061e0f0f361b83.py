def construct():
    import time, random
    from math import gcd
    start = time.time()
    DEADLINE = 80.0
    N = 128

    def lcm(a, b):
        return a * b // gcd(a, b)

    lcm_tab = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            lcm_tab[i][j] = lcm(i, j)

    conflict = [[set() for _ in range(N + 1)] for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            li = lcm_tab[i][j]
            for k in range(1, N + 1):
                if k == i or k == j:
                    continue
                if lcm_tab[i][k] == li and lcm_tab[j][k] == li:
                    conflict[i][j].add(k); conflict[j][i].add(k)
                    conflict[k][i].add(j); conflict[k][j].add(i)

    def can_add(Sset, x, S):
        for i in S:
            if conflict[i][x] & Sset:
                return False
        return True

    def greedy(order):
        S = []; Sset = set()
        for x in order:
            if x in Sset:
                continue
            if can_add(Sset, x, S):
                S.append(x); Sset.add(x)
        return S

    tot = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            tot[i] += len(conflict[i][j])

    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: tot[x]),
        sorted(range(1, N + 1), key=lambda x: -tot[x]),
    ]
    best = None
    for o in orders:
        S = greedy(o)
        if best is None or len(S) > len(best):
            best = S

    def local(S):
        S = set(S)
        improved = True
        while improved and (time.time() - start) < DEADLINE:
            improved = False
            for x in range(1, N + 1):
                if x in S:
                    continue
                if can_add(S, x, list(S)):
                    S.add(x); improved = True; break
            if improved:
                continue
            for i in list(S):
                if (time.time() - start) > DEADLINE:
                    break
                S2 = S - {i}
                for x in range(1, N + 1):
                    if x in S2:
                        continue
                    if not can_add(S2, x, list(S2)):
                        continue
                    S2b = S2 | {x}
                    ext = True
                    while ext:
                        ext = False
                        for y in range(1, N + 1):
                            if y in S2b:
                                continue
                            if can_add(S2b, y, list(S2b)):
                                S2b.add(y); ext = True; break
                    if len(S2b) > len(S):
                        S = S2b; improved = True
                        break
                if improved:
                    break
        return S

    best = local(set(best))
    rng = random.Random(42)
    while (time.time() - start) < DEADLINE - 3:
        order = list(range(1, N + 1)); rng.shuffle(order)
        S = local(set(greedy(order)))
        if len(S) > len(best):
            best = S
    if not best:
        best = [1]
    return list(best)
