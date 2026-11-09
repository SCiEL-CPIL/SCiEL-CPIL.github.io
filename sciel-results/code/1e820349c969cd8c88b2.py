# EVOLVE-BLOCK-START
import numpy as np


def ring_routing() -> np.ndarray:
    """
    Solve the ring loading problem on a ring of n = 10 nodes with 10 edges,
    routing 15 demands either clockwise or counter-clockwise to minimize
    the maximum edge load.
    
    Uses a vectorized exhaustive search over all 2^15 possible routing configurations.
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
    # cw_mask[i, j] is 1 if edge j is on the clockwise path of demand i.
    cw_masks = np.zeros((D, n), dtype=np.int64)
    for i, (s, t, w) in enumerate(demands):
        if s != t:
            length = (t - s) % n
            for k in range(length):
                edge_idx = (s + k) % n
                cw_masks[i, edge_idx] = 1

    weights = np.array([w for s, t, w in demands], dtype=np.float64)
    
    # Base load if all demands go clockwise
    base_load = cw_masks.T @ weights
    
    # Contribution of switching demand i to CCW
    # Load_j = base_load_j + sum_i x_i * w_i * (1 - 2*cw_masks[i,j])
    # diff_contrib[i, j] = w_i * (1 - 2*cw_masks[i,j])
    diff_contrib = weights[:, np.newaxis] * (1 - 2 * cw_masks)
    
    # Generate all 2^D binary vectors
    # masks: shape (2^D,)
    masks = np.arange(1 << D, dtype=np.int64)
    
    # X: shape (2^D, D) where X[k, i] is bit i of mask k
    # Use broadcasting: (2^D, 1) >> (1, D)
    X = ((masks[:, np.newaxis] >> np.arange(D)) & 1).astype(np.float64)
    
    # Calculate loads for all configurations
    # Loads shape: (2^D, n)
    # Loads[k, j] = base_load[j] + sum_i X[k, i] * diff_contrib[i, j]
    # X @ diff_contrib.T has shape (2^D, n)
    loads = base_load[None, :] + X @ diff_contrib.T
    
    # Find the configuration with the minimum maximum load
    max_loads = np.max(loads, axis=1)
    best_idx = np.argmin(max_loads)
    best_routing = X[best_idx]
    
    return best_routing

# EVOLVE-BLOCK-END
