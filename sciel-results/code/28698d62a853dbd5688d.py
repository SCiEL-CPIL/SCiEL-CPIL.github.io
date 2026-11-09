def construct():
    import time
    from math import gcd, lcm

    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    N = 255  # values 2..256
    values = list(range(2, 257))
    wnum = [256 // v for v in values]  # 1/v = wnum[v]/256

    # Compatibility: compatible[i] bit j set iff lcm(values[i], values[j]) > 256 (i != j)
    compatible = [0] * N
    for i in range(N):
        vi = values[i]
        ci = 0
        for j in range(N):
            if i == j:
                continue
            if lcm(vi, values[j]) > 256:
                ci |= (1 << j)
        compatible[i] = ci

    # Greedy weight-coloring bound on a candidate set P.
    # Returns (bound_num, color_of) where bound = sum over color classes of max weight in class.
    def color_bound(P):
        remaining = P
        color_of = {}
        bound = 0
        color = 1
        while remaining:
            # choose a vertex with max weight in this class
            best = -1
            best_w = -1
            r = remaining
            while r:
                lsb = r & -r
                idx = lsb.bit_length() - 1
                if wnum[idx] > best_w:
                    best_w = wnum[idx]
                    best = idx
                r ^= lsb
            bound += best_w
            color_of[best] = color
            remaining &= ~(1 << best)
            remaining &= ~compatible[best]  # class = independent set (no edges)
            color += 1
        return bound, color_of

    best_sum = 0
    best_mask = 0

    # Greedy initial feasible solution (valid clique in compatibility graph)
    avail = [True] * N
    R = 0
    gsum = 0
    for _ in range(N):
        bi = -1
        bw = -1
        for i in range(N):
            if avail[i] and wnum[i] > bw:
                bw = wnum[i]
                bi = i
        if bi == -1:
            break
        R |= (1 << bi)
        gsum += bw
        for i in range(N):
            if i == bi:
                avail[i] = False
            elif not (compatible[bi] >> i) & 1:
                avail[i] = False
    best_sum = gsum
    best_mask = R

    all_mask = (1 << N) - 1
    # order candidates by weight descending
    order = sorted(range(N), key=lambda i: -wnum[i])
    order_mask = 0
    for i in order:
        order_mask |= (1 << i)

    # Bron-Kerbosch with pivot + coloring bound
    def bk(Rm, Pm, Xm, cur):
        nonlocal best_sum, best_mask
        if time.time() > deadline:
            return
        if not Pm:
            if cur > best_sum:
                best_sum = cur
                best_mask = Rm
            return

        bound, color_of = color_bound(Pm)
        if cur + bound <= best_sum:
            return

        # pivot: max degree in P
        pivot = -1
        max_deg = -1
        t = Pm
        while t:
            lsb = t & -t
            u = lsb.bit_length() - 1
            d = (Pm & compatible[u]).bit_count()
            if d > max_deg:
                max_deg = d
                pivot = u
            t ^= lsb

        if pivot == -1:
            return

        candidates = Pm & ~compatible[pivot]
        candidates &= order_mask  # iterate in weight order

        t = candidates
        while t:
            lsb = t & -t
            v = lsb.bit_length() - 1
            if cur + wnum[v] > best_sum:
                bk(Rm | lsb, Pm & compatible[v], Xm & compatible[v], cur + wnum[v])
            Pm &= ~lsb
            Xm |= lsb
            if time.time() > deadline:
                return
            t ^= lsb

    bk(0, all_mask, 0, 0)

    # Build result
    res = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        res.append(values[idx])
        m ^= lsb
    if not res:
        res = [2]

    # Verify; fallback to greedy if invalid
    ok = True
    for i in range(len(res)):
        for j in range(i + 1, len(res)):
            if lcm(res[i], res[j]) <= 256:
                ok = False
                break
        if not ok:
            break
    if not ok:
        res2 = []
        m = R
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            res2.append(values[idx])
            m ^= lsb
        if res2:
            res = res2
        else:
            res = [2]
    return res
