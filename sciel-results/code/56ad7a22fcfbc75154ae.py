import numpy as np
import time
from itertools import combinations

def ring_routing() -> np.ndarray:
    N = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    t0 = time.time()
    deadline = 620.0  # safety margin before 630s kill

    # Build edge-usage matrices
    A_cw = np.zeros((D, N), dtype=np.float64)
    weights = np.zeros(D, dtype=np.float64)
    for i, (s, t, w) in enumerate(demands):
        weights[i] = w
        cur = s
        while cur != t:
            A_cw[i, cur] = 1.0
            cur = (cur + 1) % N
    A_ccw = 1.0 - A_cw

    W = weights
    L_cw = W @ A_cw  # baseline load if all CW
    M = (W[:, np.newaxis]) * (A_ccw - A_cw)  # M[i,e] = weight_i * (ccw_use - cw_use)

    # Full exhaustive search: 2^15 = 32768 combos
    # Vectorized: build all binary vectors
    num_combos = 1 << D
    powers = 1 << np.arange(D)
    ints = np.arange(num_combos, dtype=np.int64)
    AllR = ((ints[:, np.newaxis] // powers) % 2).astype(np.float64)

    # Compute all loads
    AllLoads = L_cw[None, :] + AllR @ M
    MaxLoads = np.max(AllLoads, axis=1)
    best_idx = int(np.argmin(MaxLoads))
    best_routing = AllR[best_idx].copy()
    best_max_load = float(MaxLoads[best_idx])

    # The exhaustive search already found the global optimum.
    # Verify and return.
    elapsed = time.time() - t0
    if elapsed > deadline:
        return best_routing

    return best_routing
