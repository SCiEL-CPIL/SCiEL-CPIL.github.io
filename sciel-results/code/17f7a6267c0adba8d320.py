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

    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit

    # Brute force all 2^15 combinations - guarantees global optimum
    best_max_load = np.inf
    best_routing = np.zeros(D, dtype=np.int64)

    # Vectorized approach: process all masks at once in chunks
    total_masks = 1 << D  # 32768
    chunk_size = 4096

    for chunk_start in range(0, total_masks, chunk_size):
        if time.time() > deadline:
            break
        chunk_end = min(chunk_start + chunk_size, total_masks)
        # Create array of masks for this chunk
        masks = np.arange(chunk_start, chunk_end, dtype=np.int64)
        # Convert to binary vectors: shape (chunk_len, D)
        bit_positions = np.arange(D, dtype=np.int64)
        x = ((masks[:, np.newaxis] >> bit_positions) & 1).astype(np.int64)
        # Compute loads: shape (chunk_len, n)
        loads = base_load[np.newaxis, :] + x @ contrib
        # Find max load per routing
        max_loads = loads.max(axis=1)
        # Find best in this chunk
        local_best_idx = np.argmin(max_loads)
        local_best_load = max_loads[local_best_idx]
        if local_best_load < best_max_load:
            best_max_load = local_best_load
            best_routing = x[local_best_idx]

    return best_routing.astype(np.float64)


# EVOLVE-BLOCK-END
