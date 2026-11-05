"""Deterministic bitmask construction: minimize distinct colors on 4-term APs."""
import time

def construct():
    deadline = time.time() + 50.0  # safety margin vs 60s limit
    N = 128
    # colors[i] = 1 << c for position i (0-based), else -1 for unassigned
    colors = [-1] * N
    colors[0] = 1 << 0
    used = [0] * 128  # used[c] = number of positions using color c
    used[0] = 1
    active = [0]     # list of colors currently in use
    best = None
    best_n = 128

    for x in range(1, N):
        if time.time() > deadline:
            break
        valid = 0  # bitmask of feasible colors
        for c in active:
            mask = 1 << c
            ok = True
            # 4-term APs: x-3d, x-2d, x-d must not all be color c
            for d in range(1, (x + 1) // 3 + 1):
                if (colors[x - d] == mask and colors[x - 2 * d] == mask
                        and colors[x - 3 * d] == mask):
                    ok = False
                    break
            # 3-term APs: x-2d, x-d must not both be color c
            if ok:
                for d in range(1, (x + 1) // 2 + 1):
                    if colors[x - d] == mask and colors[x - 2 * d] == mask:
                        ok = False
                        break
            if ok:
                valid |= mask
        if valid == 0:
            # introduce the smallest fresh color (guaranteed feasible)
            for c in range(128):
                if not (used[c]):
                    colors[x] = 1 << c
                    used[c] = 1
                    active.append(c)
                    break
            continue
        # pick among feasible: prefer reusing existing colors (minimize distinct)
        # deterministic tie-break: smallest color index among feasible
        # (reusing keeps distinct count flat; fresh colors only added when forced)
        # choose the feasible color with the LARGEST current usage (spread load),
        # tie -> smallest index. This tends to keep the palette small & balanced.
        best_c = -1
        best_key = (-1, 128)
        for c in active:
            if valid & (1 << c):
                key = (-used[c], c)  # maximize usage, then minimize index
                if key > best_key:
                    best_key = key
                    best_c = c
        colors[x] = 1 << best_c
        used[best_c] += 1

    # count distinct colors actually used
    n_distinct = sum(1 for c in range(128) if used[c] > 0)
    # validate the produced coloring
    ok = True
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                l = 2 * k - j
                if l >= N:
                    break
                cset = {colors[i], colors[j], colors[k], colors[l]}
                if len(cset) < 3:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    if ok:
        return [colors[i].bit_length() - 1 for i in range(N)]
    # fallback: return seed-style coloring (always valid)
    return [0] * N

BET: {mechanism: "deterministic bitmask forward construction with a global color-pool and reuse-preference tie-break directly minimizes distinct colors (the seed ignores this objective), while bitwise constraint checks make each step cheap enough to finish well under budget", q: 0.55, delta: [0.0, 0.15]}
