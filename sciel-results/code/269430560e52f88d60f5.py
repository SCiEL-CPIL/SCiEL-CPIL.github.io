import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solves the ring routing problem by exhaustive search over all 2^15 routing configurations
    using Gray code for O(1) incremental load updates per step.
    Returns the optimal routing as a float array (1.0 for CCW, 0.0 for CW).
    """
    n = 10
    demands = [
        (0, 5, 3),
        (1, 7, 2),
        (2, 9, 4),
        (3, 6, 1),
        (0, 4, 5),
        (2, 8, 2),
        (5, 9, 3),
        (1, 4, 2),
        (6, 0, 4),
        (3, 8, 1),
        (7, 2, 3),
        (4, 9, 2),
        (8, 1, 5),
        (0, 6, 1),
        (5, 2, 2),
    ]
    d = len(demands)

    # Pre-compute CW and CCW edge usage matrices (d, n)
    cw = np.zeros((d, n), dtype=np.int64)
    ccw = np.zeros((d, n), dtype=np.int64)
    w = np.zeros(d, dtype=np.int64)

    for i, (s, t, wt) in enumerate(demands):
        w[i] = wt
        if s == t:
            continue
        e = s
        while e != t:
            cw[i, e] = 1
            e = (e + 1) % n
        ccw[i] = 1 - cw[i]

    # Base load if all demands go CW
    base_load = np.sum(w[:, None] * cw, axis=0)  # shape (n,)

    # Delta load if demand i switches from CW to CCW
    delta = w[:, None] * (ccw - cw)  # shape (d, n)

    best_max_load = np.inf
    best_config = np.zeros(d, dtype=np.int64)

    # Iterate over all 2^15 configurations using Gray code
    current_load = base_load.copy()
    current_config = np.zeros(d, dtype=np.int64)

    # Pre-compute Gray code transition table for speed
    # For each mask, we need to know which bit changes from previous mask
    # and whether it turns on or off

    total = 1 << d
    for mask in range(total):
        if mask == 0:
            current_max = np.max(current_load)
            if current_max < best_max_load:
                best_max_load = current_max
                best_config = current_config.copy()
            continue

        # Gray code for current and previous mask
        gray_curr = mask ^ (mask >> 1)
        gray_prev = (mask - 1) ^ ((mask - 1) >> 1)

        # Find the bit that changed
        diff = gray_curr ^ gray_prev
        bit_idx = diff.bit_length() - 1

        # Determine if we turned ON (CW->CCW) or OFF (CCW->CW)
        if gray_curr & diff:
            # Turned ON: x_i became 1. Add delta[i]
            current_load += delta[bit_idx]
            current_config[bit_idx] = 1
        else:
            # Turned OFF: x_i became 0. Subtract delta[i]
            current_load -= delta[bit_idx]
            current_config[bit_idx] = 0

        current_max = np.max(current_load)
        if current_max < best_max_load:
            best_max_load = current_max
            best_config = current_config.copy()

    return best_config.astype(np.float64)
