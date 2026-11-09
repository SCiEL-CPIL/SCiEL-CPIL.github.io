import time
import random
import math

def construct():
    t_start = time.time()
    DEADLINE = t_start + 75.0
    N = 128
    MAXC = 128

    # Precompute all nonconstant 4-term progressions (0-indexed)
    aps = []
    for d in range(1, N // 3 + 1):
        lim = N - 3 * d
        for a in range(0, lim):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    M = len(aps)
    pos_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for p in ap:
            pos_aps[p].append(idx)

    def make_state(colors):
        apdc = [0] * M
        viol = 0
        c = colors
        for idx, ap in enumerate(aps):
            dc = len({c[i] for i in ap})
            apdc[idx] = dc
            if dc < 3:
                viol += 1
        cnt = [0] * MAXC
        for v in c:
            cnt[v] += 1
        distinct = sum(1 for v in cnt if v > 0)
        return [c, apdc, viol, cnt, distinct]

    def flip(st, p, new):
        c = st[0]
        old = c[p]
        if old == new:
            return
        cnt = st[3]
        distinct = st[4]
        if cnt[old] == 1:
            distinct -= 1
        cnt[old] -= 1
        if cnt[new] == 0:
            distinct += 1
        cnt[new] += 1
        c[p] = new
        apdc = st[1]
        viol = st[2]
        for idx in pos_aps[p]:
            ap = aps[idx]
            od = apdc[idx]
            dc = len({c[i] for i in ap})
            apdc[idx] = dc
            if od < 3 and dc >= 3:
                viol -= 1
            elif od >= 3 and dc < 3:
                viol += 1
        st[2] = viol
        st[4] = distinct

    # Guaranteed-valid greedy fallback (seed-like)
    def greedy():
        colors = []
        for x in range(1, N + 1):
            for color in range(MAXC):
                ok = True
                for d in range(1, (x - 1) // 3 + 1):
                    s = {color}
                    for j in (1, 2, 3):
                        s.add(colors[x - 1 - j * d])
                    if len(s) < 3:
                        ok = False
                        break
                if not ok:
                    continue
                for d in range(1, (x - 1) // 2 + 1):
                    if not any(colors[x - 1 - j * d] != color for j in (1, 2)):
                        ok = False
                        break
                if ok:
                    colors.append(color)
                    break
        return colors

    best_colors = greedy()
    st0 = make_state(best_colors)
    best_viol = st0[2]
    best_distinct = st0[4]
    assert best_viol == 0

    def sa(palette, init_colors, seed, iters):
        rng = random.Random(seed)
        if init_colors is not None:
            colors = init_colors[:]
        else:
            colors = [rng.choice(palette) for _ in range(N)]
        st = make_state(colors)
        cur_obj = st[2] * 10000 + st[4]
        best_c = colors[:]
        best_obj = cur_obj
        T = 500.0
        for it in range(iters):
            if time.time() > DEADLINE:
                break
            p = rng.randrange(N)
            new = rng.choice(palette)
            old = st[0][p]
            if new == old:
                continue
            ov = st[2]
            od = st[4]
            old_obj = ov * 10000 + od
            flip(st, p, new)
            obj = st[2] * 10000 + st[4]
            if obj <= old_obj or rng.random() < math.exp(-max(0, obj - old_obj) / T):
                if obj < best_obj:
                    best_obj = obj
                    best_c = st[0][:]
                T *= 0.99997
            else:
                flip(st, p, old)
        return best_c, best_obj

    inits3 = [
        lambda i: i % 3,
        lambda i: (i // 2) % 3,
        lambda i: (i * i) % 3,
        lambda i: (i * 7 + 3) % 3,
        lambda i: (i * 5 + 1) % 3,
        None,
    ]

    seed = 98765
    iters_per = 4000
    restart = 0
    # Strategy cycle weighted toward palette-3 from-scratch and refinement
    strategy_cycle = ["p3", "p3", "refine", "p3", "p4", "p3", "p5", "refine", "p3", "p4"]
    si = 0
    while time.time() < DEADLINE - 1.5:
        strat = strategy_cycle[si % len(strategy_cycle)]
        si += 1
        if strat == "refine" and best_viol == 0:
            pal = sorted(set(best_colors))
            while len(pal) < 3:
                pal.append(len(pal))
            bc, _ = sa(pal, best_colors, seed, iters_per)
            seed += 1
            stx = make_state(bc)
            if stx[2] == 0 and stx[4] < best_distinct:
                best_colors = bc
                best_viol = 0
                best_distinct = stx[4]
        else:
            if strat == "p3":
                init = inits3[restart % len(inits3)]
                initc = [init(i) for i in range(N)] if init is not None else None
                bc, _ = sa([0, 1, 2], initc, seed, iters_per)
                seed += 1
                stx = make_state(bc)
                if stx[2] == 0 and stx[4] < best_distinct:
                    best_colors = bc
                    best_viol = 0
                    best_distinct = stx[4]
            elif strat == "p4":
                bc, _ = sa([0, 1, 2, 3], None, seed, iters_per)
                seed += 1
                stx = make_state(bc)
                if stx[2] == 0 and stx[4] < best_distinct:
                    best_colors = bc
                    best_viol = 0
                    best_distinct = stx[4]
            elif strat == "p5":
                bc, _ = sa([0, 1, 2, 3, 4], None, seed, iters_per)
                seed += 1
                stx = make_state(bc)
                if stx[2] == 0 and stx[4] < best_distinct:
                    best_colors = bc
                    best_viol = 0
                    best_distinct = stx[4]
        restart += 1
        if best_distinct <= 3:
            # Optimal lower bound reached; brief extra refine then stop
            pal = sorted(set(best_colors))
            while len(pal) < 3:
                pal.append(len(pal))
            bc, _ = sa(pal, best_colors, seed, iters_per)
            seed += 1
            stx = make_state(bc)
            if stx[2] == 0 and stx[4] < best_distinct:
                best_colors = bc
                best_distinct = stx[4]
            break

    # Final safety: ensure validity
    stf = make_state(best_colors)
    if stf[2] != 0:
        best_colors = greedy()
    return list(best_colors)
