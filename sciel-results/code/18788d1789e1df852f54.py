# EVOLVE-BLOCK-START
import time
import numpy as np

def ring_routing() -> np.ndarray:
    t_start = time.time()
    DEADLINE = 630.0
    SAFETY = 10.0

    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    D = len(demands)

    cw = np.zeros((D, n), dtype=np.int8)
    ccw = np.zeros((D, n), dtype=np.int8)
    for i, (s, t, w) in enumerate(demands):
        d = (t - s) % n
        for k in range(d):
            cw[i, (s + k) % n] = 1
        for k in range(d, n):
            ccw[i, (s + k) % n] = 1

    # Order demands by descending weight for tighter branch-and-bound pruning.
    order = sorted(range(D), key=lambda i: -demands[i][2])
    cw_o = cw[order]
    ccw_o = ccw[order]

    best = np.full(n, np.inf)
    best_load = np.inf
    best_assign = None

    def consider(load):
        nonlocal best, best_load, best_assign
        m = load.max()
        if m < best_load:
            best_load = m
            best = load.copy()
            a = np.zeros(D, dtype=np.int8)
            for pos, di in enumerate(order):
                a[di] = 1
            best_assign = a

    def branch(pos, load):
        nonlocal best_load
        if time.time() > DEADLINE - SAFETY:
            return
        if pos == D:
            m = load.max()
            if m < best_load:
                best_load = m
                best = load.copy()
                a = np.zeros(D, dtype=np.int8)
                for p, di in enumerate(order):
                    a[di] = 1
                best_assign = a
            return
        # Prune: even if all remaining demands are free, current max is a lower bound.
        if load.max() >= best_load:
            return
        cw_l = cw_o[pos]
        ccw_l = ccw_o[pos]
        # Try counter-clockwise (1) first, then clockwise (0)
        load += ccw_l
        branch(pos + 1, load)
        load -= ccw_l
        load += cw_l
        branch(pos + 1, load)
        load -= cw_l

    branch(0, np.zeros(n, dtype=np.int64))

    # Simulated-annealing polish to escape any local optimum.
    if best_assign is not None and time.time() < DEADLINE - SAFETY:
        cur = best_assign.copy()
        cur_load = np.zeros(n, dtype=np.int64)
        for i in range(D):
            cur_load += (ccw[i] if cur[i] else cw[i])
        cur_m = cur_load.max()
        best_sa = cur.copy()
        best_sa_m = cur_m
        T = 2.0
        for it in range(400000):
            if time.time() > DEADLINE - SAFETY:
                break
            i = np.random.randint(D)
            new_dir = 1 - cur[i]
            if new_dir:
                delta = ccw[i] - cw[i]
            else:
                delta = cw[i] - ccw[i]
            new_load = cur_load + delta
            new_m = new_load.max()
            if new_m < cur_m or (new_m == cur_m and np.random.rand() < 0.5) or (np.random.rand() < np.exp(-(new_m - cur_m) / max(T, 1e-6))):
                cur[i] = new_dir
                cur_load = new_load
                cur_m = new_m
                if cur_m < best_sa_m:
                    best_sa_m = cur_m
                    best_sa = cur.copy()
            T *= 0.9999

        if best_sa_m < best_load:
            best_load = best_sa_m
            best_assign = best_sa.copy()

    if best_assign is None:
        return np.zeros(D)
    return best_assign.astype(np.float64)

# EVOLVE-BLOCK-END
