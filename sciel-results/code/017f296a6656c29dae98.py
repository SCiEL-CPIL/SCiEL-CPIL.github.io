# EVOLVE-BLOCK-START
import numpy as np
import time


def ring_routing() -> np.ndarray:
    """
    Solve the ring loading problem on a ring of n = 10 nodes with 10 edges,
    routing 15 demands either clockwise or counter-clockwise to minimize
    the maximum edge load.
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
    D = len(demands)

    # Precompute clockwise edge masks for each demand.
    cw_masks = []
    for s, t, w in demands:
        mask = np.zeros(n, dtype=np.int64)
        if s != t:
            for i in range((t - s) % n):
                edge_idx = (s + i) % n
                mask[edge_idx] = 1
        cw_masks.append(mask)

    cw_masks = np.array(cw_masks, dtype=np.int64)  # shape (D, n)
    weights = np.array([w for s, t, w in demands], dtype=np.int64)  # shape (D,)
    ccw_masks = 1 - cw_masks  # shape (D, n)

    base_load = cw_masks.T @ weights  # shape (n,)
    diff_masks = ccw_masks - cw_masks  # shape (D, n)
    contrib = weights[:, np.newaxis] * diff_masks  # shape (D, n)

    # Brute force all 2^15 combinations - guarantees global optimum
    # Since D=15, 2^15 = 32768, which is small enough for a single vectorized pass
    total_masks = 1 << D
    masks = np.arange(total_masks, dtype=np.int64)
    bit_positions = np.arange(D, dtype=np.int64)
    
    # Convert to binary vectors: shape (total_masks, D)
    x_all = ((masks[:, np.newaxis] >> bit_positions) & 1).astype(np.int64)
    
    # Compute loads for all routings: shape (total_masks, n)
    all_loads = base_load[np.newaxis, :] + x_all @ contrib
    
    # Find max load per routing
    max_loads = all_loads.max(axis=1)
    
    # Find the best routing
    best_idx = np.argmin(max_loads)
    best_routing = x_all[best_idx]
    
    return best_routing.astype(np.float64)


# EVOLVE-BLOCK-END
