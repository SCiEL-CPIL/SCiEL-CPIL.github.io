# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import time

def ring_routing() -> np.ndarray:
    """
    Exact brute-force over all 2^15 routing assignments on a 10-node ring.
    Returns the routing that minimises the maximum edge load.
    """
    N = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    weights = np.array([d[2] for d in demands], dtype=np.int64)

    # Precompute clockwise edge bitmasks for each demand
    cw_mask = np.zeros(D, dtype=np.int64)
    for i, (s, t, w) in enumerate(demands):
        m = 0
        e = s
        while e != t:
            m |= (1 << e)
            e = (e + 1) % N
        cw_mask[i] = m

    all_edges_mask = (1 << N) - 1
    ccw_mask = all_edges_mask & ~cw_mask

    # Precompute contribution arrays: for each demand, weight added to each edge
    # cw_contrib[i, e] = weights[i] if e in cw_mask[i] else 0
    cw_contrib = np.zeros((D, N), dtype=np.int64)
    ccw_contrib = np.zeros((D, N), dtype=np.int64)
    for i in range(D):
        for e in range(N):
            if (cw_mask[i] >> e) & 1:
                cw_contrib[i, e] = weights[i]
            if (ccw_mask[i] >> e) & 1:
                ccw_contrib[i, e] = weights[i]

    # Brute force: 2^15 = 32768 assignments
    best_routing = np.zeros(D, dtype=np.float64)
    best_max_load = np.inf

    t0 = time.time()
    deadline = t0 + 580.0  # safety margin

    for mask in range(1 << D):
        if time.time() > deadline:
            break
        # For each demand, if bit is set -> CCW (1.0), else CW (0.0)
        loads = np.zeros(N, dtype=np.int64)
        for i in range(D):
            if (mask >> i) & 1:
                loads += ccw_contrib[i]
            else:
                loads += cw_contrib[i]
        ml = int(np.max(loads))
        if ml < best_max_load:
            best_max_load = ml
            best_routing = np.array([(mask >> i) & 1 for i in range(D)], dtype=np.float64)

    return best_routing

# EVOLVE-BLOCK-END
