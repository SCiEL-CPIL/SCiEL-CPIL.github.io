# EVOLVE-BLOCK-START
import numpy as np
import time


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        0 -> route the demand clockwise
        1 -> route the demand counter-clockwise (value >= 0.5)

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    D = 15
    N = 10

    # Demands: (source, target, weight)
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

    weights = np.array([d[2] for d in demands], dtype=float)

    # Precompute incidence matrices for CW and CCW paths
    # cw_matrix[i, e] = 1 if edge e is in CW path of demand i
    cw_matrix = np.zeros((D, N), dtype=float)
    ccw_matrix = np.zeros((D, N), dtype=float)
    
    for i, (s, t, w) in enumerate(demands):
        e = s
        while e != t:
            cw_matrix[i, e] = 1.0
            e = (e + 1) % N
        # CCW is the complement
        ccw_matrix[i] = 1.0 - cw_matrix[i]

    # Total number of combinations: 2^15 = 32768
    # Exhaustive search using Gray code for efficiency
    
    # Safety margin before 600s limit
    deadline = time.time() + 590

    best_load = float('inf')
    best_routing = np.zeros(D)

    # Initialize loads with all clockwise
    # loads = sum(weights[i] * cw_matrix[i] for i in range(D))
    # Vectorized: (cw_matrix.T @ weights)
    loads = cw_matrix.T @ weights

    current_max = np.max(loads)
    if current_max < best_load:
        best_load = current_max
        best_routing = np.zeros(D)

    # Track current routing state
    state = np.zeros(D, dtype=int)  # 0 = cw, 1 = ccw

    for i in range(1, 1 << D):
        if time.time() > deadline:
            break

        # Compute gray code for i
        g = i ^ (i >> 1)
        # Previous gray code
        g_prev = (i - 1) ^ ((i - 1) >> 1)
        # Find the bit that changed
        diff = g ^ g_prev
        bit = diff.bit_length() - 1

        # Update state and loads
        if state[bit] == 0:
            # Switching from CW to CCW
            # Net change: weights[bit] * (ccw_matrix[bit] - cw_matrix[bit])
            loads += weights[bit] * (ccw_matrix[bit] - cw_matrix[bit])
            state[bit] = 1
        else:
            # Switching from CCW to CW
            # Net change: weights[bit] * (cw_matrix[bit] - ccw_matrix[bit])
            loads += weights[bit] * (cw_matrix[bit] - ccw_matrix[bit])
            state[bit] = 0

        current_max = np.max(loads)
        if current_max < best_load:
            best_load = current_max
            best_routing = state.copy()

    return best_routing


# EVOLVE-BLOCK-END
