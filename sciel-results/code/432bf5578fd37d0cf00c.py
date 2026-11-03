# EVOLVE-BLOCK-START
import time
import numpy as np


def _greedy(a, b, n, rng):
    # Build a placement by inserting facilities (highest total flow first) at the
    # location that minimizes the incremental cost, given the facilities already placed.
    order = np.argsort(-a.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[:] = -1
    used = np.zeros(n, dtype=bool)
    B = np.zeros((n, n), dtype=np.int64)  # B[i, loc] = a[i][p[loc]]
    for idx in order:
        inc = B[idx, :] * (b.sum(axis=1) + b.sum(axis=0))
        inc[used] = np.inf
        loc = int(np.argmin(inc))
        p[loc] = idx
        used[loc] = True
        B[idx, loc] = a[idx, loc]
    return p, B


def _cost(a, b, p):
    return int(np.sum(a * b[np.arange(len(p)), p]))


def solve(a, b):
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    n = len(a)
    deadline = time.time() + 140.0
    rng = np.random.default_rng(0)

    best_p, best_B = _greedy(a, b, n, rng)
    best_cost = _cost(a, b, best_p)
    rowsums = a.sum(axis=1)

    while time.time() < deadline:
        p = best_p.copy()
        B = best_B.copy()
        cur = best_cost
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # --- 2-opt swap pass ---
            for x in range(n):
                if time.time() > deadline:
                    break
                bx = B[:, x]
                for y in range(x + 1, n):
                    by = B[:, y]
                    delta = 2 * np.dot(rowsums, bx + by)
                    if delta < 0:
                        p[x], p[y] = p[y], p[x]
                        B[:, x], B[:, y] = by, bx
                        cur += delta
                        improved = True
            # --- 3-opt cycle pass (needs full B3 = a @ B) ---
            if time.time() < deadline:
                B3 = a @ B
                for x in range(n):
                    if time.time() > deadline:
                        break
                    for y in range(x + 1, n):
                        for z in range(y + 1, n):
                            delta = (B3[x, y] + B3[y, z] + B3[z, x]
                                     - B3[x, x] - B3[y, y] - B3[z, z])
                            if delta < 0:
                                p[x], p[y], p[z] = p[y], p[z], p[x]
                                B[:, x], B[:, y], B[:, z] = B[:, y], B[:, z], B[:, x]
                                cur += delta
                                improved = True
        if cur < best_cost:
            best_cost = cur
            best_p = p.copy()
            best_B = B.copy()
        else:
            # no improvement this round: restart from a fresh greedy build for diversity
            p2, B2 = _greedy(a, b, n, rng)
            c2 = _cost(a, b, p2)
            if c2 < best_cost:
                best_cost = c2
                best_p = p2
                best_B = B2
            else:
                # no progress at all -> stop
                break

    return best_p.tolist()
# EVOLVE-BLOCK-END
