# EVOLVE-BLOCK-START
import time
import numpy as np


def _build_data():
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    D = len(demands)
    cw = []
    ccw = []
    for (s, t, w) in demands:
        if s == t:
            cw_edges = []
        else:
            cw_edges = []
            e = s
            while e != t:
                cw_edges.append(e)
                e = (e + 1) % n
        ccw_edges = [e for e in range(n) if e not in set(cw_edges)]
        cw.append((cw_edges, w))
        ccw.append((ccw_edges, w))
    return demands, cw, ccw, n, D


def ring_routing() -> np.ndarray:
    start = time.time()
    deadline = start + 600.0
    safety = 30.0

    demands, cw, ccw, n, D = _build_data()

    # Precompute per-demand edge lists and weights
    cw_edges = [cw[i][0] for i in range(D)]
    ccw_edges = [ccw[i][0] for i in range(D)]
    weights = np.array([demands[i][2] for i in range(D)], dtype=np.float64)

    # Precompute delta matrix: delta[i][e] = change in load on edge e
    # when demand i switches from CW to CCW
    # = +w_i if e in ccw_i, -w_i if e in cw_i, 0 otherwise
    delta = np.zeros((D, n), dtype=np.float64)
    for i in range(D):
        w = weights[i]
        for e in cw_edges[i]:
            delta[i, e] = -w
        for e in ccw_edges[i]:
            delta[i, e] = w

    # Initial loads: all clockwise
    loads = np.zeros(n, dtype=np.float64)
    for i in range(D):
        for e in cw_edges[i]:
            loads[e] += weights[i]

    best_load = float(np.max(loads))
    best_gray = 0  # all zeros = all CW

    # Gray code enumeration over all 2^D routings
    total = 1 << D
    prev_gray = 0
    for step in range(1, total):
        gray = step ^ (step >> 1)
        diff = gray ^ prev_gray
        bit = (diff & -diff).bit_length() - 1
        is_ccw = (gray >> bit) & 1

        # Update loads using precomputed delta vector
        if is_ccw:
            loads += delta[bit]
        else:
            loads -= delta[bit]

        prev_gray = gray
        cur_max = float(np.max(loads))
        if cur_max < best_load:
            best_load = cur_max
            best_gray = gray

        if time.time() > deadline - safety:
            break

    # Construct routing from best_gray
    routing = np.zeros(D)
    for i in range(D):
        routing[i] = 1.0 if (best_gray >> i) & 1 else 0.0

    return routing


# EVOLVE-BLOCK-END
