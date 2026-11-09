# EVOLVE-BLOCK-START
import numpy as np
import time


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges.
    Uses an exhaustive search over all 2^15 possible routing combinations to find
    the one that minimizes the maximum edge load.
    """
    start_time = time.time()
    deadline = start_time + 585.0  # Safety margin for 630s limit

    n_nodes = 10
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

    # Precompute the edge sets for clockwise and counter-clockwise routes
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        # Clockwise: edges s, s+1, ..., t-1 (mod n)
        cw = set()
        e = s
        while e != t:
            cw.add(e)
            e = (e + 1) % n_nodes
        cw_edges.append(cw)
        # Counter-clockwise: all other edges
        all_edges = set(range(n_nodes))
        ccw = all_edges - cw
        ccw_edges.append(ccw)

    # Convert to boolean matrices for fast vectorized operations
    cw_matrix = np.zeros((D, n_nodes), dtype=np.float64)
    ccw_matrix = np.zeros((D, n_nodes), dtype=np.float64)
    for i in range(D):
        for e in cw_edges[i]:
            cw_matrix[i, e] = 1.0
        for e in ccw_edges[i]:
            ccw_matrix[i, e] = 1.0

    weights = np.array([w for _, _, w in demands], dtype=np.float64)

    # Precompute base loads (all clockwise) and difference matrix
    base_loads = weights @ cw_matrix
    diff_matrix = ccw_matrix - cw_matrix  # shape (D, n_nodes)

    # Precompute all 2^15 routing vectors as a (32768, D) matrix
    # routing_vecs[m, i] = 1 if demand i is routed CCW in mask m
    n_masks = 1 << D
    routing_vecs = np.zeros((n_masks, D), dtype=np.float64)
    for mask in range(n_masks):
        for i in range(D):
            routing_vecs[mask, i] = (mask >> i) & 1

    # Precompute weighted routing vectors: weighted_routing[mask, i] = weights[i] * routing[mask, i]
    # Then loads[mask] = base_loads + diff_matrix.T @ (weights * routing[mask])
    # = base_loads + diff_matrix.T @ weighted_routing[mask]
    weighted_routing = routing_vecs * weights  # (n_masks, D)

    best_max_load = float('inf')
    best_mask = 0

    # Process in chunks to be memory-friendly and check deadline
    chunk_size = 4096
    for chunk_start in range(0, n_masks, chunk_size):
        if time.time() > deadline:
            break
        chunk_end = min(chunk_start + chunk_size, n_masks)
        chunk_weighted = weighted_routing[chunk_start:chunk_end]  # (chunk_size, D)
        
        # Compute loads for all masks in this chunk
        # loads = base_loads + diff_matrix.T @ chunk_weighted.T  -> shape (n_nodes, chunk_size)
        chunk_loads = base_loads[:, np.newaxis] + diff_matrix.T @ chunk_weighted.T  # (n_nodes, chunk_size)
        chunk_max_loads = np.max(chunk_loads, axis=0)  # (chunk_size,)
        
        local_best_idx = np.argmin(chunk_max_loads)
        local_best_load = chunk_max_loads[local_best_idx]
        
        if local_best_load < best_max_load:
            best_max_load = local_best_load
            best_mask = chunk_start + local_best_idx

    # Reconstruct best routing
    best_routing = np.array([(best_mask >> i) & 1 for i in range(D)], dtype=np.float64)
    return best_routing


# EVOLVE-BLOCK-END
