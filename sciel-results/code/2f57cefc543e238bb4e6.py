def construct():
    N = 128
    best = []
    best_size = 0

    def try_grow(seed, order):
        S = sorted(seed)
        m = set(S)
        for x in order:
            if x in m:
                continue
            if _can_add(S, m, x):
                S.append(x)
                S.sort()
                m.add(x)
        return S

    allnums = list(range(1, N+1))
    orders = [
        list(range(1, N+1)),
        list(range(N, 0, -1)),
    ]

    for L in range(1, N+1):
        seed = list(range(L, N+1))
        for order in orders:
            S = try_grow(seed, order)
            if len(S) > best_size:
                best_size = len(S)
                best = S

    # Also try empty seed
    for order in orders:
        S = try_grow([], order)
        if len(S) > best_size:
            best_size = len(S)
            best = S

    # final verify
    assert _valid(best)
    # ensure python ints, list
    return [int(x) for x in best]
