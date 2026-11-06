import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before the 150s kill

    # ------------------------------------------------------------------
    # Bitset K2,2,2 detector.
    # K2,2,2 (complete tripartite, parts of size 2) is a non-induced
    # subgraph of colour c iff there exist three disjoint pairs
    # A={a1,a2}, B={b1,b2}, C={c1,c2} such that:
    #   - within each pair the edge is NOT colour c,
    #   - all 12 cross-edges between the three pairs ARE colour c.
    # ------------------------------------------------------------------
    def has_k222(colors, n):
        if n < 6:
            return False
        adj0 = [0] * n
        adj1 = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                idx += 1
                if c == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                else:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)

        for c in (0, 1):
            adj = adj0 if c == 0 else adj1
            for i in range(n):
                ai = adj[i]
                for j in range(i + 1, n):
                    if ai & (1 << j):
                        continue  # i,j adjacent in colour c -> cannot share a part
                    S_mask = ai & adj[j] & ~(1 << i) & ~(1 << j)
                    # S = common colour-c neighbours of i and j (the other parts)
                    # For each non-c pair (b1,b2) in S, we need a non-c pair
                    # (c1,c2) in S that is fully colour-c connected to b1,b2.
                    m = S_mask
                    S = []
                    while m:
                        v = (m & -m).bit_length() - 1
                        S.append(v)
                        m &= m - 1
                    s_len = len(S)
                    if s_len < 4:
                        continue
                    for bi in range(s_len):
                        b1 = S[bi]
                        ab1 = adj[b1]
                        for bj in range(bi + 1, s_len):
                            b2 = S[bj]
                            if ab1 & (1 << b2):
                                continue  # b1,b2 adjacent in c -> cannot share a part
                            cand = S_mask & ab1 & adj[b2] & ~(1 << b1) & ~(1 << b2)
                            mc = cand
                            C = []
                            while mc:
                                v = (mc & -mc).bit_length() - 1
                                C.append(v)
                                mc &= mc - 1
                            clen = len(C)
                            for ci in range(clen):
                                c1 = C[ci]
                                ac1 = adj[c1]
                                for cj in range(ci + 1, clen):
                                    c2 = C[cj]
                                    if not (ac1 & (1 << c2)):
                                        return True
        return False

    # ------------------------------------------------------------------
    # Paley colouring for prime n ≡ 1 (mod 4).
    # ------------------------------------------------------------------
    def paley(n):
        squares = {x * x % n for x in range(1, n)}
        return [0 if (j - i) % n in squares else 1
                for i in range(n) for j in range(i + 1, n)]

    # ------------------------------------------------------------------
    # Fast edge-flip cost delta via bitsets.
    # Flipping edge (i,j) from c to 1-c changes the number of K2,2,2 of
    # colour c by (lost - gained).
    # ------------------------------------------------------------------
    def edge_flip_delta(adj, n, i, j, c):
        # vertices that connect to BOTH i and j in colour c (exclude i,j)
        both = adj[i] & adj[j] & ~(1 << i) & ~(1 << j)
        # vertices that connect to exactly one of i,j in colour c
        xor = (adj[i] ^ adj[j]) & ~(1 << i) & ~(1 << j)

        # K2,2,2 of colour c that use the edge (i,j) as a WITHIN-PART edge
        # (i,j is not colour c): third part = a non-c pair inside `both`.
        lost_within = 0
        mb = both
        B = []
        while mb:
            v = (mb & -mb).bit_length() - 1
            B.append(v)
            mb &= mb - 1
        for a in range(len(B)):
            v1 = B[a]
            av1 = adj[v1]
            for b in range(a + 1, len(B)):
                v2 = B[b]
                if not (av1 & (1 << v2)):
                    lost_within += 1

        # K2,2,2 of colour c that use the edge (i,j) as a CROSS edge:
        # i,j in the same part; third part = a non-c pair in `xor`.
        gained = 0
        mx = xor
        X = []
        while mx:
            v = (mx & -mx).bit_length() - 1
            X.append(v)
            mx &= mx - 1
        for a in range(len(X)):
            v1 = X[a]
            av1 = adj[v1]
            for b in range(a + 1, len(X)):
                v2 = X[b]
                if not (av1 & (1 << v2)):
                    gained += 1

        return lost_within - gained  # delta in count of colour-c K2,2,2

    # ------------------------------------------------------------------
    # Local search: minimise total number of K2,2,2 (both colours).
    # ------------------------------------------------------------------
    def local_search(n, init_colors, time_limit, rng):
        adj0 = [0] * n
        adj1 = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = init_colors[idx]
                idx += 1
                if c == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                else:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)

        cur_cost = 0
        best_cost = 0
        best_adj0 = adj0[:]
        best_adj1 = adj1[:]

        def apply_flip(i, j, c):
            # flip edge (i,j) which is currently colour c, to 1-c
            bit_i = 1 << i
            bit_j = 1 << j
            if c == 0:
                adj0[i] &= ~bit_j
                adj0[j] &= ~bit_i
                adj1[i] |= bit_j
                adj1[j] |= bit_i
            else:
                adj1[i] &= ~bit_j
                adj1[j] &= ~bit_i
                adj0[i] |= bit_j
                adj0[j] |= bit_i

        t_end = time.time() + time_limit
        temp = 1.0
        while time.time() < t_end:
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            c = 0 if (adj0[i] & (1 << j)) else 1
            delta0 = edge_flip_delta(adj0, n, i, j, 0) if c == 0 else -edge_flip_delta(adj0, n, i, j, 0)
            # delta0 = change in colour-0 K2,2,2 count from this flip
            # delta1 = change in colour-1 K2,2,2 count = -delta0 (edge moves between colours)
            delta1 = -delta0
            total_delta = delta0 + delta1
            if total_delta < 0 or (temp > 0.01 and rng.random() < (2.718281828459045 ** (-total_delta / temp))):
                apply_flip(i, j, c)
                cur_cost += total_delta
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_adj0 = adj0[:]
                    best_adj1 = adj1[:]
                    if best_cost <= 0:
                        break
            temp *= 0.99
            if temp < 0.01:
                temp = 0.01

        # rebuild best colors
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(0 if (best_adj0[i] & (1 << j)) else 1)
        return colors

    # ------------------------------------------------------------------
    # Deterministic attempt: Paley 29, verify, return if valid.
    # ------------------------------------------------------------------
    for n_try in (29, 25, 21, 17):
        cols = paley(n_try)
        if not has_k222(cols, n_try):
            return {'n': n_try, 'colors': cols}

    # ------------------------------------------------------------------
    # Local-search attempts for n=29 (and fallbacks).
    # ------------------------------------------------------------------
    rng = random.Random(12345)
    for n_try in (29, 25, 21):
        if time.time() > deadline - 20:
            break
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            init = [rng.randint(0, 1) for _ in range(n_try * (n_try - 1) // 2)]
            cols = local_search(n_try, init, time_limit=25.0, rng=rng)
            if not has_k222(cols, n_try):
                return {'n': n_try, 'colors': cols}

    # ------------------------------------------------------------------
    # Fallback: smallest verified Paley (guaranteed valid).
    # ------------------------------------------------------------------
    for n_try in (17, 13, 5):
        cols = paley(n_try)
        if not has_k222(cols, n_try):
            return {'n': n_try, 'colors': cols}

    # Absolute last resort (should not happen).
    return {'n': 1, 'colors': []}
