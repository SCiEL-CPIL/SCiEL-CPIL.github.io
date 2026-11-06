def construct():
    import random
    from math import gcd
    N = 128
    nums = list(range(1, N+1))
    # Precompute gcd matrix indexed 1..N
    G = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            G[i][j] = g
            G[j][i] = g

    def is_bad_triple(a, b, c):
        return G[a][b] == G[a][c] == G[b][c]

    def can_add(x, S_set, S_list):
        # S_list is list of elements in S_set
        for idx in range(len(S_list)):
            a = S_list[idx]
            for jdx in range(idx+1, len(S_list)):
                b = S_list[jdx]
                g = G[a][b]
                if G[x][a] == g and G[x][b] == g:
                    return False
        return True

    def is_valid(S_list):
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i+1, n):
                b = S_list[j]
                g = G[a][b]
                for k in range(j+1, n):
                    c = S_list[k]
                    if G[a][c] == g and G[b][c] == g:
                        return False
        return True

    def repair(S_list):
        # Remove elements until valid (greedy removal of a participant in a bad triple)
        S_list = list(S_list)
        changed = True
        while changed:
            changed = False
            n = len(S_list)
            for i in range(n):
                a = S_list[i]
                for j in range(i+1, n):
                    b = S_list[j]
                    g = G[a][b]
                    for k in range(j+1, n):
                        c = S_list[k]
                        if G[a][c] == g and G[b][c] == g:
                            # remove c (or the "least useful"); remove c
                            S_list.remove(c)
                            changed = True
                            break
                    if changed: break
                if changed: break
        return S_list

    best = None
    best_size = 0
    rng = random.Random(12345)

    def greedy(order):
        S = []
        Sset = set()
        for x in order:
            if x in Sset:
                continue
            if can_add(x, Sset, S):
                S.append(x)
                Sset.add(x)
        return S

    def local_improve(S, iters=200):
        S = list(S)
        Sset = set(S)
        for _ in range(iters):
            improved = False
            # try adds
            for x in nums:
                if x in Sset:
                    continue
                if can_add(x, Sset, S):
                    S.append(x)
                    Sset.add(x)
                    improved = True
            if not improved:
                # try swaps: remove one in S, add one not in S, keep valid
                progressed = False
                for a in list(S):
                    for b in nums:
                        if b in Sset:
                            continue
                        # candidate new set
                        S2 = [t for t in S if t != a]
                        S2set = Sset - {a}
                        if can_add(b, S2set, S2):
                            # accept swap only if it doesn't reduce size (it keeps size)
                            # and we hope it unblocks adds; accept if new set valid (it is) 
                            # To avoid churn, accept swaps that allow at least one further add
                            S.append(b)
                            Sset.add(b)
                            S.remove(a)
                            Sset.discard(a)
                            progressed = True
                            # after swap, try adds
                            for x in nums:
                                if x in Sset:
                                    continue
                                if can_add(x, Sset, S):
                                    S.append(x)
                                    Sset.add(x)
                            break
                    if progressed:
                        break
                if not progressed:
                    break
        return S

    # Heuristic base order: numbers sorted by (number of divisors) descending? 
    # Try multiple strategies.
    strategies = []
    # strategy 1: ascending
    strategies.append(list(nums))
    # strategy 2: descending
    strategies.append(list(reversed(nums)))
    # strategy 3: by number of prime factors desc (more composite first)
    def omega(x):
        c = 0
        d = x
        while d % 2 == 0:
            c += 1; d //= 2
        p = 3
        while p*p <= d:
            while d % p == 0:
                c += 1; d //= p
            p += 2
        if d > 1: c += 1
        return c
    strategies.append(sorted(nums, key=lambda x: (-omega(x), x)))
    strategies.append(sorted(nums, key=lambda x: (omega(x), x)))
    # random orders
    for _ in range(30):
        o = list(nums)
        rng.shuffle(o)
        strategies.append(o)

    for order in strategies:
        S = greedy(order)
        S = local_improve(S)
        if not is_valid(S):
            S = repair(S)
        if len(S) > best_size:
            best_size = len(S)
            best = list(S)

    if best is None or len(best) == 0:
        best = [2]

    # Final safety: ensure valid and nonempty
    if not is_valid(best):
        best = repair(best)
    if len(best) == 0:
        best = [2]
    return best
