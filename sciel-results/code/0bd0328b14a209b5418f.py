def construct():
    import time, random
    import numpy as np
    t0 = time.time()
    DEADLINE = t0 + 50.0
    n = 128
    aps = []
    for a in range(1, n+1):
        md = (n - a)//3
        for d in range(1, md+1):
            aps.append((a-1, a+d-1, a+2*d-1, a+3*d-1))
    m = len(aps)
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for p in ap:
            pos_aps[p].append(idx)
    # Precompute arrays for vectorized ops
    ap_arr = np.array(aps, dtype=np.int32)  # shape (m, 4)
    pos_l = pos_aps
    m_l = m
    n_l = n
    aps_l = aps

    def count_distinct4(colors, a, b, c, d):
        x0 = colors[a]; x1 = colors[b]; x2 = colors[c]; x3 = colors[d]
        if x0 == x1:
            if x0 == x2:
                if x0 == x3: return 1
                return 2
            if x0 == x3: return 2
            if x2 == x3: return 2
            return 3
        else:
            if x2 == x0:
                if x3 == x0: return 2
                if x3 == x1: return 3
                return 3
            elif x2 == x1:
                if x3 == x1: return 2
                if x3 == x0: return 3
                return 3
            else:
                if x3 == x0: return 3
                if x3 == x1: return 3
                if x3 == x2: return 3
                return 4

    def is_bad(colors, ap):
        return count_distinct4(colors, ap[0], ap[1], ap[2], ap[3]) < 3

    def compute_vset(colors):
        vset = set()
        for i in range(m_l):
            ap = aps_l[i]
            if is_bad(colors, ap):
                vset.add(i)
        return vset

    def try_k(k, budget):
        end = t0 + budget
        best = None
        best_du = 10**9
        rng = random
        while time.time() < end:
            # init
            if rng.random() < 0.5:
                colors = [i % k for i in range(n_l)]
            else:
                colors = [rng.randrange(k) for _ in range(n_l)]
            vset = compute_vset(colors)
            stall = 0
            while time.time() < end and vset:
                # pick a violating edge
                ei = rng.choice(list(vset))
                ap = aps_l[ei]
                cur = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
                # candidate moves: vertex in ap, color not in cur
                cand = []
                for p in ap:
                    for c in range(k):
                        if c not in cur:
                            cand.append((p, c))
                if not cand:
                    cand = [(ap[0], rng.randrange(k))]
                # evaluate best delta
                best_delta = 10**9
                best_mv = None
                for (p, c) in cand:
                    dlt = 0
                    for idx in pos_l[p]:
                        a2 = aps_l[idx]
                        was = idx in vset
                        # compute distinct count after change
                        cd = count_distinct4(colors, a2[0], a2[1], a2[2], a2[3])
                        # need to account for p being changed
                        # rebuild
                        s = set()
                        for q in a2:
                            s.add(c if q == p else colors[q])
                        now = len(s) < 3
                        if was and not now:
                            dlt -= 1
                        elif (not was) and now:
                            dlt += 1
                    if dlt < best_delta:
                        best_delta = dlt
                        best_mv = (p, c)
                p, c = best_mv
                old = colors[p]
                colors[p] = c
                for idx in pos_l[p]:
                    a2 = aps_l[idx]
                    now = is_bad(colors, a2)
                    was = idx in vset
                    if was and not now:
                        vset.discard(idx)
                    elif (not was) and now:
                        vset.add(idx)
                if best_delta < 0:
                    stall = 0
                else:
                    stall += 1
                if stall > 800:
                    # kick
                    for _ in range(8):
                        p2 = rng.randrange(n_l)
                        c2 = rng.randrange(k)
                        if c2 == colors[p2]:
                            continue
                        colors[p2] = c2
                        for idx in pos_l[p2]:
                            a2 = aps_l[idx]
                            now = is_bad(colors, a2)
                            was = idx in vset
                            if was and not now:
                                vset.discard(idx)
                            elif (not was) and now:
                                vset.add(idx)
                    stall = 0
            if not vset:
                du = len(set(colors))
                if du < best_du:
                    best_du = du
                    best = colors[:]
                if best_du == k and k == 3:
                    return best
        return best

    # search
    result = None
    budgets = {3: 18, 4: 14, 5: 10, 6: 7, 7: 5, 8: 4}
    for k in range(3, 9):
        if time.time() > DEADLINE - 3:
            break
        b = budgets.get(k, 5)
        rem = DEADLINE - 3 - time.time()
        b = min(b, rem)
        if b <= 0.5:
            break
        res = try_k(k, b)
        if res is not None:
            ok = True
            for ap in aps:
                if len({res[ap[0]], res[ap[1]], res[ap[2]], res[ap[3]]}) < 3:
                    ok = False
                    break
            if ok:
                result = res
                break
    if result is not None:
        return result
    # fallback greedy
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors
