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
    # Clockwise from s to t uses edges s, s+1, ..., t-1 (mod n).
    cw_masks = []
    for s, t, w in demands:
        mask = np.zeros(n, dtype=np.int64)
        if s == t:
            # All edges? Actually, if s==t, clockwise path is the whole ring? 
            # The problem says "Clockwise from s to t uses edges s, s+1, ..., t-1 (mod 10)".
            # If s == t, this range is empty? Or the whole ring?
            # Let's check the definition: "counter-clockwise uses all other ring edges".
            # If s==t, clockwise path length is 0? Then counter-clockwise is all 10 edges?
            # Or is it the other way?
            # Usually, distance 0 means no edges. But let's look at the demands: none have s==t.
            # So this case doesn't occur.
            pass
        else:
            # Edges are indexed by their start node.
            # Edge i connects i to (i+1)%n.
            # Path s -> t clockwise visits nodes s, s+1, ..., t.
            # Edges used: s, s+1, ..., t-1.
            # Number of edges: (t - s) % n.
            # If (t - s) % n == 0, it's the whole ring (10 edges). But s!=t so this is 1..9.
            for i in range((t - s) % n):
                edge_idx = (s + i) % n
                mask[edge_idx] = 1
        cw_masks.append(mask)

    cw_masks = np.array(cw_masks, dtype=np.int64)  # shape (D, n)
    weights = np.array([w for s, t, w in demands], dtype=np.int64)  # shape (D,)

    # Counter-clockwise mask is the complement (all other edges).
    # Since the ring has n edges, and clockwise uses k edges, counter-clockwise uses n-k edges.
    # Note: if k=0 (s==t), cw is empty, ccw is all. If k=n, cw is all, ccw is empty.
    # For s != t, k is in 1..n-1, so both are non-empty and disjoint, covering all edges.
    ccw_masks = 1 - cw_masks  # shape (D, n)

    # We want to choose x_i in {0, 1} for each demand i.
    # x_i = 0 -> clockwise, x_i = 1 -> counter-clockwise.
    # Load on edge e = sum_i weights[i] * ( (1-x_i) * cw_masks[i,e] + x_i * ccw_masks[i,e] )
    # = sum_i weights[i] * cw_masks[i,e] + sum_i weights[i] * x_i * (ccw_masks[i,e] - cw_masks[i,e])
    
    # Let base_load[e] = sum_i weights[i] * cw_masks[i,e]  (if all routed clockwise)
    # Let delta[e] = sum_i weights[i] * x_i * (ccw_masks[i,e] - cw_masks[i,e])
    # Total load[e] = base_load[e] + delta[e]
    
    base_load = cw_masks.T @ weights  # shape (n,)
    # diff_masks[i,e] = ccw_masks[i,e] - cw_masks[i,e]
    # Since ccw_masks = 1 - cw_masks, diff = 1 - 2*cw_masks
    diff_masks = ccw_masks - cw_masks  # shape (D, n)
    
    # We want to minimize max_e (base_load[e] + sum_i weights[i] * x_i * diff_masks[i,e])
    
    # Brute force is feasible: 2^15 = 32768 combinations.
    # We can vectorize this.
    
    # Create a matrix of all possible routing vectors? 32768 x 15.
    # Or iterate through masks 0 to 2^15 - 1.
    
    best_max_load = np.inf
    best_routing = np.zeros(D, dtype=np.int64)
    
    start_time = time.time()
    deadline = start_time + 600  # Safety margin
    
    # Efficient brute force using bit manipulation and vectorized numpy
    # For each mask from 0 to 2^15 - 1:
    #   x = bits of mask
    #   load = base_load + diff_masks.T @ (weights * x)
    #   max_load = max(load)
    #   if max_load < best_max_load: update
    
    # To speed up, we can precompute the contribution of each demand to the load vector.
    # contrib[i] = weights[i] * diff_masks[i]  (shape n,)
    # Then load = base_load + sum_{i: x_i=1} contrib[i]
    
    contrib = weights[:, np.newaxis] * diff_masks  # shape (D, n)
    
    # Iterate through all 2^15 masks
    # This is 32768 iterations. Each involves a sum of up to 15 vectors of length 10.
    # 32768 * 15 * 10 = ~5 million operations. Very fast.
    
    # We can do even better by dynamic programming or Gray code, but 32k is small.
    
    # Let's just loop.
    for mask in range(1 << D):
        if time.time() > deadline:
            break
        
        # Extract bits
        # x_i = (mask >> i) & 1
        # We need the sum of contrib[i] for all i where bit i is set.
        
        # Efficient way:
        # sum_vec = np.zeros(n)
        # for i in range(D):
        #     if (mask >> i) & 1:
        #         sum_vec += contrib[i]
        
        # Even faster: precompute sums for chunks of bits?
        # Or just use numpy where.
        
        x = np.array([(mask >> i) & 1 for i in range(D)], dtype=np.int64)
        # load = base_load + x @ contrib
        load = base_load + x @ contrib
        max_load = np.max(load)
        
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = x
    
    # Convert routing to the required format:
    # Entry >= 0.5 routes counter-clockwise (x=1), else clockwise (x=0).
    # The function returns an array of 15 numbers.
    # x is 0 or 1. So we can just return x as float.
    
    return best_routing.astype(np.float64)


# EVOLVE-BLOCK-END
