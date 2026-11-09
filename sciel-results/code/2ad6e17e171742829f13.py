# EVOLVE-BLOCK-START
import numpy as np
import time


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.
    """
    N = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    # Precompute CW edge masks as bitmasks (10 bits, bit i = edge i)
    cw_masks = []
    for s, t, w in demands:
        mask = 0
        edge = s
        while edge != t:
            mask |= (1 << edge)
            edge = (edge + 1) % N
        cw_masks.append(mask)
    
    # CCW mask is complement within 10 bits
    all_edges_mask = (1 << N) - 1  # 1023
    ccw_masks = [all_edges_mask ^ m for m in cw_masks]
    
    # Brute force all 2^15 = 32768 combinations
    # For each combination, compute edge loads and find max
    best_max_load = float('inf')
    best_routing = np.zeros(D)
    
    start_time = time.time()
    deadline = start_time + 600  # safety margin
    
    for combo in range(1 << D):
        if time.time() > deadline - 5:
            break
        
        edge_loads = np.zeros(N, dtype=np.int32)
        for i in range(D):
            if combo & (1 << i):
                # CCW
                mask = ccw_masks[i]
                w = demands[i][2]
                # Add weight to each set bit in mask
                m = mask
                while m:
                    lsb = m & (-m)
                    edge_idx = lsb.bit_length() - 1
                    edge_loads[edge_idx] += w
                    m ^= lsb
            else:
                # CW
                mask = cw_masks[i]
                w = demands[i][2]
                m = mask
                while m:
                    lsb = m & (-m)
                    edge_idx = lsb.bit_length() - 1
                    edge_loads[edge_idx] += w
                    m ^= lsb
        
        max_load = np.max(edge_loads)
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = np.array([(1.0 if (combo >> i) & 1 else 0.0) for i in range(D)])
    
    return best_routing


# EVOLVE-BLOCK-END
