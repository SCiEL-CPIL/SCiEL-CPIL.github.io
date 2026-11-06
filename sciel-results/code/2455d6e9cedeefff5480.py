def construct():
    import time
    start = time.time()
    DEADLINE = 78.0  # safety margin before 90s kill

    N = 128
    nums = list(range(1, N + 1))
    idx = {n: i for i, n in enumerate(nums)}  # 0..127

    # prime exponent signature per number
    def prime_exp_sig(n):
        d = {}
        m = n
        p = 2
        while p * p <= m:
            if m % p == 0:
                e = 0
                while m % p == 0:
                    m //= p
                    e += 1
                d[p] = e
            p += 1 if p == 2 else 2
        if m > 1:
            d[m] = d.get(m, 0) + 1
        return d

    sigs = [prime_exp_sig(n) for n in nums]

    def is_bad(a, b, c):
        # a,b,c are indices; check triple bad
        # for every prime in union, at least two attain the max exponent
        primes = set(sigs[a].keys()) | set(sigs[b].keys()) | set(sigs[c].keys())
        for p in primes:
            ea = sigs[a].get(p, 0)
            eb = sigs[b].get(p, 0)
            ec = sigs[c].get(p, 0)
            mx = max(ea, eb, ec)
            cnt = (1 if ea == mx else 0) + (1 if eb == mx else 0) + (1 if ec == mx else 0)
            if cnt < 2:
                return False
        return True

    # pair_conflict[u][w] = bitmask of v != u,w with {u,v,w} bad
    pair_conflict = [[0] * N for _ in range(N)]
    for u in range(N):
        for w in range(u + 1, N):
            mask = 0
            for v in range(N):
                if v == u or v == w:
                    continue
                if is_bad(u, v, w):
                    mask |= (1 << v)
            pair_conflict[u][w] = mask
            pair_conflict[w][u] = mask

    # greedy lower bound with a good ordering
    def greedy(order):
        chosen = []
        chosen_set = set()
        for v in order:
            ok = True
            for i in range(len(chosen)):
                for j in range(i + 1, len(chosen)):
                    if is_bad(chosen[i], chosen[j], v):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                chosen.append(v)
                chosen_set.add(v)
        return chosen

    # ordering: try a few
    orders = [
        list(range(N)),
        list(range(N - 1, -1, -1)),
        sorted(range(N), key=lambda i: (nums[i].bit_length(), nums[i])),
        sorted(range(N), key=lambda i: -len(sigs[i])),
    ]
    best = []
    for o in orders:
        g = greedy(o)
        if len(g) > len(best):
            best = g

    best_mask = 0
    for v in best:
        best_mask |= (1 << v)
    best_size = len(best)

    # B&B
    import sys
    sys.setrecursionlimit(100000)

    # find a bad triple in candidate mask C, return (u,w,v) or None
    def find_bad_triple(C):
        # iterate set bits
        # collect indices
        idxs = []
        tmp = C
        while tmp:
            lsb = tmp & (-tmp)
            i = lsb.bit_length() - 1
            idxs.append(i)
            tmp ^= lsb
        L = len(idxs)
        for i in range(L):
            u = idxs[i]
            for j in range(i + 1, L):
                w = idxs[j]
                m = pair_conflict[u][w] & C
                # remove u,w from m (they aren't in pair_conflict anyway since v!=u,w)
                if m:
                    lsb = m & (-m)
                    v = lsb.bit_length() - 1
                    return (u, w, v)
        return None

    best_solution = best_mask
    best_size_found = best_size
    nodes = [0]

    def popcount(x):
        return bin(x).count('1')

    def branch(C, depth):
        nonlocal best_solution, best_size_found
        nodes[0] += 1
        if nodes[0] % 20000 == 0:
            if time.time() > DEADLINE:
                return
        ub = popcount(C)
        if ub <= best_size_found:
            return
        t = find_bad_triple(C)
        if t is None:
            # C is independent
            if ub > best_size_found:
                best_size_found = ub
                best_solution = C
            return
        u, w, v = t
        # branch exclude each
        # order branches: exclude the one that reduces most? just try all three
        for x in (u, w, v):
            C2 = C & ~(1 << x)
            if popcount(C2) > best_size_found:
                branch(C2, depth + 1)
                if time.time() > DEADLINE:
                    return

    branch(best_mask, 0)

    # also try B&B from full set to potentially find larger
    if time.time() < DEADLINE - 5:
        full = (1 << N) - 1
        branch(full, 0)

    # build result from best_solution
    result = [nums[i] for i in range(N) if (best_solution >> i) & 1]
    if not result:
        result = [1]
    return result
