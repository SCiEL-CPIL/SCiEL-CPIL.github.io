# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solves the ring loading problem by finding the routing that minimizes the maximum edge load.
    Uses exhaustive search over all 2^15 possible routing combinations, which is feasible and guarantees the global optimum.
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
    # mask[d, e] = 1 if demand d uses edge e when routed clockwise.
    cw_masks = np.zeros((D, n), dtype=np.int64)
    for d, (s, t, w) in enumerate(demands):
        if s != t:
            length = (t - s) % n
            for i in range(length):
                edge_idx = (s + i) % n
                cw_masks[d, edge_idx] = 1
    
    weights = np.array([w for s, t, w in demands], dtype=np.int64)
    
    # If we route demand d clockwise, it contributes weights[d] * cw_masks[d] to loads.
    # If we route demand d counter-clockwise, it contributes weights[d] * (1 - cw_masks[d]) to loads.
    
    # Let x_d be 1 if routed clockwise, 0 if counter-clockwise.
    # Load on edge e = sum_d weights[d] * (x_d * cw_masks[d, e] + (1-x_d) * (1-cw_masks[d, e]))
    # Load_e = sum_d weights[d] * (1 - cw_masks[d, e]) + sum_d weights[d] * x_d * (2*cw_masks[d, e] - 1)
    
    # Base load if all routed counter-clockwise (x_d = 0)
    # ccw_mask = 1 - cw_masks
    ccw_masks = 1 - cw_masks
    base_load = ccw_masks.T @ weights  # shape (n,)
    
    # Delta load if we switch demand d from ccw to cw
    # delta_d = weights[d] * (cw_masks[d] - ccw_masks[d]) = weights[d] * (2*cw_masks[d] - 1)
    diff_masks = 2 * cw_masks - 1  # shape (D, n)
    contrib = weights[:, np.newaxis] * diff_masks  # shape (D, n)
    
    # Total load for a given routing vector x (binary, shape (D,)) is:
    # loads = base_load + x @ contrib
    
    # We need to find x in {0, 1}^D that minimizes max(loads)
    
    # Since D=15, we can iterate all 2^15 = 32768 possibilities.
    # We can vectorize this by creating a matrix of all possible x vectors.
    
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # Generate all possible routing vectors
    # We can use bit manipulation to generate indices 0 to 2^D - 1 and convert to binary.
    total_combinations = 1 << D
    indices = np.arange(total_combinations, dtype=np.int64)
    
    # Convert indices to binary vectors of shape (total_combinations, D)
    # Bit i of index corresponds to demand i.
    # Note: np.unpackbits might be slow or require specific padding.
    # Simple loop or matrix multiplication approach is better.
    
    # Create a matrix where row i is the binary representation of i.
    # We can do this by shifting right and masking.
    bit_positions = np.arange(D, dtype=np.int64)
    # indices[:, np.newaxis] >> bit_positions
    # This creates an array of shape (total_combinations, D)
    x_all = (indices[:, np.newaxis] >> bit_positions) & 1
    x_all = x_all.astype(np.int64)
    
    # Compute loads for all combinations at once
    # loads_all shape: (total_combinations, n)
    loads_all = base_load[np.newaxis, :] + x_all @ contrib
    
    # Find the max load for each combination
    max_loads = loads_all.max(axis=1)
    
    # Find the combination with the minimum max load
    best_idx = np.argmin(max_loads)
    best_routing = x_all[best_idx]
    
    # Return as float64 as requested
    return best_routing.astype(np.float64)

# EVOLVE-BLOCK-END
