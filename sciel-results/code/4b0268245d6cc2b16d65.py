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
        1 -> route the demand counter-clockwise

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

    # Precompute the set of edges for each demand in each direction.
    # Clockwise from s to t uses edges s, s+1, ..., t-1 (mod N).
    # Counter-clockwise uses all other edges.
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        cw = set()
        e = s
        while e != t:
            cw.add(e)
            e = (e + 1) % N
        ccw = set(range(N)) - cw
        cw_edges.append(cw)
        ccw_edges.append(ccw)

    # Total number of combinations: 2^15 = 32768
    # We can do an exact exhaustive search using Gray code for efficiency.

    deadline = time.time() + 590  # Safety margin before 600s limit

    best_load = float('inf')
    best_routing = np.zeros(D)

    # Use Gray code enumeration to efficiently compute loads
    # Start with all zeros (all clockwise)
    # Gray code: g = i ^ (i >> 1)
    # Consecutive Gray codes differ in exactly one bit.

    # Initialize loads with all clockwise
    loads = np.zeros(N)
    for i in range(D):
        for e in cw_edges[i]:
            loads[e] += demands[i][2]

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

        # Update state
        if state[bit] == 0:
            # Switching from CW to CCW: subtract CW edges, add CCW edges
            w = demands[bit][2]
            for e in cw_edges[bit]:
                loads[e] -= w
            for e in ccw_edges[bit]:
                loads[e] += w
            state[bit] = 1
        else:
            # Switching from CCW to CW: subtract CCW edges, add CW edges
            w = demands[bit][2]
            for e in ccw_edges[bit]:
                loads[e] -= w
            for e in cw_edges[bit]:
                loads[e] += w
            state[bit] = 0

        current_max = np.max(loads)
        if current_max < best_load:
            best_load = current_max
            best_routing = state.copy()

    # If we finished early, we found the global optimum
    # Return the best routing found
    return best_routing


# EVOLVE-BLOCK-END
