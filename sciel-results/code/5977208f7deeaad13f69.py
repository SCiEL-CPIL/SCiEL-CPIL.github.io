# EVOLVE-BLOCK-START
import numpy as np


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    This implementation uses an exhaustive search (brute force) over all 2^15
    possible routing combinations to guarantee the optimal solution.

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
        0 -> route the demand clockwise
        1 -> route the demand counter-clockwise
    """
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Precompute incidence matrices for CW and CCW routing
    # M_cw[i, e] = weight of demand i if edge e is on its CW path, else 0
    # M_ccw[i, e] = weight of demand i if edge e is on its CCW path, else 0
    M_cw = np.zeros((D, n))
    M_ccw = np.zeros((D, n))
    
    for i, (s, t, w) in enumerate(demands):
        # Determine CW edges
        cw_edge_set = set()
        e = s
        while e != t:
            cw_edge_set.add(e)
            e = (e + 1) % n
        
        # Assign weights to matrices
        for e in range(n):
            if e in cw_edge_set:
                M_cw[i, e] = w
            else:
                M_ccw[i, e] = w

    best_max_load = float('inf')
    best_routing = np.zeros(D)

    # Exhaustive search over all 2^D combinations
    # D=15, so 32768 iterations. Very fast.
    for mask in range(1 << D):
        # Construct routing vector from mask
        # routing[i] = 1 if bit i is set, else 0
        # We can do this efficiently or just loop. Loop is fine for 15 bits.
        routing = np.zeros(D)
        for i in range(D):
            if (mask >> i) & 1:
                routing[i] = 1.0
        
        # Calculate loads
        # loads = sum(routing[i] * M_ccw[i] + (1-routing[i]) * M_cw[i]) for all i
        # Vectorized:
        # routing is (D,), M_ccw is (D, n)
        # loads = (routing * M_ccw).sum(axis=0) + ((1 - routing) * M_cw).sum(axis=0)
        # Or simpler:
        # loads = M_cw.sum(axis=0) + (routing * (M_ccw - M_cw)).sum(axis=0)
        
        loads = np.sum(routing[:, np.newaxis] * M_ccw + (1 - routing[:, np.newaxis]) * M_cw, axis=0)
        
        current_max = np.max(loads)
        
        if current_max < best_max_load:
            best_max_load = current_max
            best_routing = routing.copy()
            
            # Optional early termination if we find a perfect solution?
            # Theoretical minimum max load is at least total_weight / 10?
            # Total weight = 3+2+4+1+5+2+3+2+4+1+3+2+5+1+2 = 40
            # Avg load = 4. So if we find max_load == 4, it's optimal.
            if best_max_load == 4.0:
                break

    return best_routing


# EVOLVE-BLOCK-END
