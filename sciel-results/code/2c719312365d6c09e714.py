# EVOLVE-BLOCK-START
import numpy as np

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Uses exhaustive search with vectorized load calculation to guarantee the
    globally optimal solution.
    """
    n_nodes = 10
    n_edges = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Precompute CW and CCW weight matrices
    # cw_matrix[d, e] is weight of demand d on edge e if routed CW
    # ccw_matrix[d, e] is weight of demand d on edge e if routed CCW
    cw_matrix = np.zeros((D, n_edges))
    ccw_matrix = np.zeros((D, n_edges))

    for d, (s, t, w) in enumerate(demands):
        if s == t:
            continue
        # Clockwise from s to t: edges s, s+1, ..., t-1 (mod n)
        if s < t:
            cw_edges = range(s, t)
        else:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))
        
        for e in cw_edges:
            cw_matrix[d, e] = w
            
        # CCW uses the other edges
        # Since it's a ring, if CW uses a set of edges, CCW uses the complement.
        # We can just mark the rest.
        mask = cw_matrix[d] > 0
        ccw_mask = ~mask
        ccw_matrix[d, ccw_mask] = w

    # Base load if all routed CW
    Base = np.sum(cw_matrix, axis=0) # Shape (10,)
    
    # Delta load if we switch demand d from CW to CCW
    # Load_e = Base_e + sum_d x_d * (CCW_d,e - CW_d,e)
    Delta = ccw_matrix - cw_matrix # Shape (15, 10)

    best_max_load = float('inf')
    best_routing = np.zeros(D)

    # Iterate over all 2^15 possibilities
    # Using bit manipulation to generate binary vectors
    for k in range(1 << D):
        # Convert k to binary vector of length D
        # x[d] is 1 if bit d of k is set
        # We need x @ Delta + Base
        # Efficient way:
        # x is a boolean array.
        x = np.array([(k >> i) & 1 for i in range(D)], dtype=np.float64)
        
        # Calculate loads on each edge
        loads = Base + x @ Delta
        
        current_max = np.max(loads)
        
        if current_max < best_max_load:
            best_max_load = current_max
            best_routing = x

    # Ensure output format: 0 for CW, 1 for CCW (or >=0.5 for CCW)
    # The task says: "An entry >= 0.5 routes the demand counter-clockwise, any other value clockwise."
    # Our best_routing has 0.0 or 1.0.
    return best_routing

# EVOLVE-BLOCK-END
