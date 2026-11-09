# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
        0 -> clockwise, 1 -> counter-clockwise
    """
    n = 10
    D = 15

    # Demands (source, target, weight) in order
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

    # Precompute edge indices for clockwise and counter-clockwise routes
    cw_edges = []
    ccw_edges = []

    for (s, t, w) in demands:
        # Clockwise edges: s, (s+1)%n, ..., (t-1)%n
        cw = []
        if s <= t:
            for e in range(s, t):
                cw.append(e)
        else:
            for e in range(s, n):
                cw.append(e)
            for e in range(0, t):
                cw.append(e)
        cw_edges.append(cw)

        # Counter-clockwise edges: all edges not in clockwise
        all_edges = list(range(n))
        ccw = [e for e in all_edges if e not in cw]
        ccw_edges.append(ccw)

    weights = np.array([d[2] for d in demands], dtype=np.float64)

    # Precompute contribution vectors: for each demand, the edge load vector if routed CW or CCW
    cw_contrib = np.zeros((D, n), dtype=np.float64)
    ccw_contrib = np.zeros((D, n), dtype=np.float64)

    for d in range(D):
        w = weights[d]
        for e in cw_edges[d]:
            cw_contrib[d, e] = w
        for e in ccw_edges[d]:
            ccw_contrib[d, e] = w

    # Precompute: for each demand, the delta vector (change in edge loads) when flipping from CW to CCW
    # delta[d] = ccw_contrib[d] - cw_contrib[d]
    delta = ccw_contrib - cw_contrib  # shape (D, n)

    # Base load: all demands routed clockwise
    base_load = np.sum(cw_contrib, axis=0)  # shape (n,)

    # Precompute the load for every possible mask (2^15 = 32768)
    # This allows O(1) lookup during the search phase, avoiding repeated matrix multiplications
    num_masks = 1 << D
    # Use a list of numpy arrays for easy indexing and mutation during precomputation
    # Or a single 2D array. A 2D array is better for cache locality if we access sequentially.
    # However, we need to build it recursively.
    # load_table[mask] = load_table[mask without lowest set bit] + delta[lowest set bit index]
    
    # Initialize table
    load_table = np.zeros((num_masks, n), dtype=np.float64)
    load_table[0] = base_load

    # Precompute lowest set bit index for each mask to speed up recursion
    # lsb_index[mask] = index of the lowest set bit in mask
    # We can compute this on the fly or precompute.
    # For 32768 masks, on-the-fly is fine.
    
    for mask in range(1, num_masks):
        # Find lowest set bit
        # (mask & -mask) gives the lowest set bit as a power of 2
        lsb = mask & -mask
        # Index of the lowest set bit
        # We can use bit_length() - 1
        i = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        load_table[mask] = load_table[prev_mask] + delta[i]

    # Now search for the mask with the minimum maximum load
    best_max_load = float('inf')
    best_mask = 0

    # Vectorized max over the second axis gives an array of max loads for all masks
    # max_loads = np.max(load_table, axis=1)
    # This is extremely fast in NumPy.
    max_loads = np.max(load_table, axis=1)
    
    # Find the index (mask) with the minimum max load
    best_mask = np.argmin(max_loads)
    best_max_load = max_loads[best_mask]

    # Build the routing array from best_mask
    routing = np.zeros(D, dtype=np.float64)
    for d in range(D):
        if (best_mask >> d) & 1:
            routing[d] = 1.0

    return routing

# EVOLVE-BLOCK-END
