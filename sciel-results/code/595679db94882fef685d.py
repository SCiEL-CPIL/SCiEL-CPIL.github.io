# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solves the ring loading problem for 10 nodes and 15 demands.
    Uses exhaustive search over all 2^15 routing combinations to guarantee the global optimum.
    """
    # Safety margin: 630s limit, use 600s for logic
    deadline = time.time() + 590.0

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
    D = len(demands)
    weights = np.array([d[2] for d in demands], dtype=np.float64)

    # Precompute clockwise edge-usage matrix
    # cw[d, e] = 1 if demand d uses edge e clockwise
    cw = np.zeros((D, n), dtype=np.float64)
    for i, (s, t, w) in enumerate(demands):
        if s <= t:
            for e in range(s, t):
                cw[i, e] = 1.0
        else:
            for e in range(s, n):
                cw[i, e] = 1.0
            for e in range(0, t):
                cw[i, e] = 1.0

    # Counter-clockwise uses the complementary edges
    ccw = 1.0 - cw

    # Load calculation:
    # load[e] = sum_d weight[d] * (x[d] * ccw[d,e] + (1-x[d]) * cw[d,e])
    #         = sum_d weight[d] * cw[d,e] + sum_d weight[d] * x[d] * (ccw[d,e] - cw[d,e])
    base_load = cw.T @ weights  # Load if all demands are routed clockwise
    delta = ccw - cw            # Change in edge usage when switching d from CW to CCW
    w_delta = delta * weights[:, np.newaxis]

    total_combos = 1 << D
    best_max_load = np.inf
    best_routing = np.zeros(D, dtype=np.float64)

    # Process in chunks to manage memory and allow time checks
    chunk_size = 4096
    for start in range(0, total_combos, chunk_size):
        if time.time() > deadline:
            break
        
        end = min(start + chunk_size, total_combos)
        size = end - start
        indices = np.arange(start, end, dtype=np.int64)

        # Extract bits: x_d = (index >> d) & 1
        # indices shape: (size, 1)
        # bits shape: (size, D)
        bits = ((indices[:, np.newaxis] >> np.arange(D)[np.newaxis, :]) & 1).astype(np.float64)

        # Compute loads for all combinations in this chunk
        # loads shape: (size, n)
        loads = base_load[np.newaxis, :] + bits @ w_delta

        # Find the combination with the minimum maximum load in this chunk
        max_loads = loads.max(axis=1)
        local_best_idx = np.argmin(max_loads)
        
        if max_loads[local_best_idx] < best_max_load:
            best_max_load = max_loads[local_best_idx]
            best_routing = bits[local_best_idx].copy()

    return best_routing

# EVOLVE-BLOCK-END
