import time
import numpy as np


def construct():
    start = time.time()
    deadline = start + 82  # safety margin before the 90s wall-clock kill

    n = 128

    # Precompute every nonconstant 4-term AP (0-indexed): (a, a+d, a+2d, a+3d)
    aps_list = []
    for d in range(1, n):
        for a in range(0, n - 3 * d):
            aps_list.append((a, a + d, a + 2 * d, a + 3 * d))
    aps = np.array(aps_list, dtype=np.int32)

    def violation_count(c):
        ca = c[aps[:, 0]]
        cb = c[aps[:, 1]]
        cc = c[aps[:, 2]]
        cd = c[aps[:, 3]]
        d2 = 1 + (cb != ca).astype(np.int32)
        d3 = d2 + ((cc != ca) & (cc != cb)).astype(np.int32)
        d4 = d3 + ((cd != ca) & (cd != cb) & (cd != cc)).astype(np.int32)
        return int(np.sum(d4 < 3))

    def count_colors(c):
        return len(set(int(x) for x in c))

    BIG = 100000

    def cost(c):
        return BIG * violation_count(c) + count_colors(c)

    def greedy(max_colors):
        c = [-1] * n
        for x in range(n):
            placed = False
            for color in range(max_colors):
                ok = True
                for d in range(1, x // 3 + 1):
                    a = x - 3 * d
                    if a < 0:
                        break
                    vals = (c[a], c[a + d], c[a + 2 * d], color)
                    if len(set(vals)) < 3:
                        ok = False
                        break
                if ok:
                    c[x] = color
                    placed = True
                    break
            if not placed:
                return None
        return c

    # Initial valid coloring via greedy (fewest colors greedy can place)
    best = None
    for mc in range(3, 129):
        g = greedy(mc)
        if g is not None:
            best = np.array(g, dtype=np.int32)
            break
    if best is None:
        best = np.array(list(range(n)), dtype=np.int32)

    best_cost = cost(best)

    def try_improve(c, time_limit, seed):
        c = c.copy()
        current_cost = cost(c)
        best_c = c.copy()
        best_local = current_cost
        T = 100.0
        rng = np.random.default_rng(seed)
        steps = 0
        while time.time() < time_limit:
            x = int(rng.integers(n))
            old = c[x]
            new = int(rng.integers(128))
            if new == old:
                continue
            c[x] = new
            new_cost = cost(c)
            delta = new_cost - current_cost
            if delta <= 0 or (T > 1e-9 and rng.random() < np.exp(-delta / T)):
                current_cost = new_cost
                if new_cost < best_local:
                    best_local = new_cost
                    best_c = c.copy()
            else:
                c[x] = old
            steps += 1
            if steps % 300 == 0:
                T *= 0.985
        return best_c, best_local

    remaining = deadline - time.time()
    if remaining > 4:
        n_restarts = max(1, int(remaining // 8))
        per = remaining / n_restarts - 0.5
        for r in range(n_restarts):
            if time.time() > deadline - 2:
                break
            if r == 0:
                c0 = best.copy()
            else:
                c0 = best.copy()
                rng2 = np.random.default_rng(1000 + r)
                for _ in range(8):
                    x = int(rng2.integers(n))
                    c0[x] = int(rng2.integers(128))
            c1, c1_cost = try_improve(c0, time.time() + per, seed=100 + r)
            if c1_cost < best_cost:
                best = c1
                best_cost = c1_cost

    # Guarantee validity; fall back to greedy-valid if SA left us invalid
    if best_cost >= BIG:
        for mc in range(3, 129):
            g = greedy(mc)
            if g is not None:
                best = np.array(g, dtype=np.int32)
                break

    return [int(x) for x in best]
