# EVOLVE-BLOCK-START
import numpy as np
import time


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges.
    Uses an exhaustive search over all 2^15 possible routing combinations to find
    the one that minimizes the maximum edge load.
    Uses Gray-code enumeration for incremental load updates.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin before 630s limit

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

    weights = np.array([w for _, _, w in demands], dtype=np.float64)

    # Precompute the load delta for each demand if switched from CW to CCW
    # delta_loads[i, e] = weight_i * (1 if e in ccw_i and e not in cw_i else -1 if e in cw_i and e not in ccw_i else 0)
    # Actually, it's simpler:
    # load_cw[i, e] = weight_i if e in cw_i else 0
    # load_ccw[i, e] = weight_i if e in ccw_i else 0
    # If we start with all CW, loads = sum of load_cw
    # If we switch demand i to CCW, we subtract load_cw[i] and add load_ccw[i]
    # So delta[i] = load_ccw[i] - load_cw[i]

    cw_matrix = np.zeros((D, n_nodes), dtype=np.float64)
    ccw_matrix = np.zeros((D, n_nodes), dtype=np.float64)
    for i in range(D):
        for e in cw_edges[i]:
            cw_matrix[i, e] = weights[i]
        for e in ccw_edges[i]:
            ccw_matrix[i, e] = weights[i]

    # Initial loads: all demands routed CW
    loads = cw_matrix.sum(axis=0)  # shape (n_nodes,)
    
    # Delta for switching demand i from CW to CCW
    delta = ccw_matrix - cw_matrix  # shape (D, n_nodes)

    best_max_load = float('inf')
    best_mask = 0
    current_max_load = np.max(loads)
    
    # Gray code enumeration
    # gray = i ^ (i >> 1)
    # The bit that changes between gray(i-1) and gray(i) is the position of the least significant 1 in i
    # Let's iterate through all masks using Gray code
    
    n_masks = 1 << D
    
    for i in range(1, n_masks):
        if time.time() > deadline:
            break
        
        # Find which bit changes
        # The bit that changes in the Gray code sequence from i-1 to i is the position of the trailing zeros in i
        # Actually, the bit that changes is the position of the least significant set bit in i
        # Let's verify: 
        # i=1 (001): gray(0)=000, gray(1)=001. Bit 0 changes.
        # i=2 (010): gray(1)=001, gray(2)=011. Bit 1 changes.
        # i=3 (011): gray(2)=011, gray(3)=010. Bit 0 changes.
        # i=4 (100): gray(3)=010, gray(4)=110. Bit 2 changes.
        # So the bit that changes is the position of the least significant set bit in i.
        
        # Compute the bit position
        # trailing_zeros = (i & -i).bit_length() - 1
        lsb = i & -i
        bit_pos = lsb.bit_length() - 1
        
        # Determine if the bit is being set or cleared
        # In Gray code, we need to know the state of bit_pos in gray(i)
        # gray(i) = i ^ (i >> 1)
        gray_i = i ^ (i >> 1)
        if (gray_i >> bit_pos) & 1:
            # Bit is set in gray(i), meaning we switched from 0 to 1 (CW to CCW)
            # So we add delta[bit_pos] to loads
            loads += delta[bit_pos]
        else:
            # Bit is cleared in gray(i), meaning we switched from 1 to 0 (CCW to CW)
            # So we subtract delta[bit_pos] from loads
            loads -= delta[bit_pos]
        
        current_max_load = np.max(loads)
        
        if current_max_load < best_max_load:
            best_max_load = current_max_load
            best_mask = gray_i
    
    # Reconstruct best routing
    best_routing = np.array([(best_mask >> i) & 1 for i in range(D)], dtype=np.float64)
    return best_routing


# EVOLVE-BLOCK-END
