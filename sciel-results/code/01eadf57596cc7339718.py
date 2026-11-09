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
        >= 0.5 -> route the demand counter-clockwise
        < 0.5  -> route the demand clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one entry per demand.
    """
    start_time = time.time()
    deadline = 600.0

    # Demands: (source, target, weight)
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    n_edges = n_nodes

    # Precompute clockwise edge masks and weights
    # For each demand, we store:
    # 1. The set of edges used if routed clockwise (as a bitmask or list)
    # 2. The weight of the demand
    # 3. The set of edges used if routed counter-clockwise (complement of CW edges)

    # To make it fast, we can precompute the load contribution vectors.
    # However, the inspiration suggests using bitmasks and incremental updates.
    # With 10 edges, we can use a 10-bit integer for each edge set.
    
    cw_masks = [0] * n_demands
    ccw_masks = [0] * n_demands
    weights = [0] * n_demands

    for i, (s, t, w) in enumerate(demands):
        mask = 0
        current = s
        while current != t:
            mask |= (1 << current)
            current = (current + 1) % n_nodes
        
        cw_masks[i] = mask
        ccw_masks[i] = (~mask) & ((1 << n_nodes) - 1)
        weights[i] = w

    # Total number of combinations is 2^15 = 32768
    # We can enumerate all of them efficiently.
    
    best_max_load = float('inf')
    best_routing = np.zeros(n_demands)

    # Gray code enumeration to minimize changes between steps
    # But for 32768 steps, even simple enumeration with vectorized load calculation
    # or simple loops is fast enough.
    # Let's try a simple loop first, but optimized.
    
    # Actually, 32768 iterations is tiny. We can just loop.
    # To be safe and fast, we can use numpy for the load calculation in each step?
    # No, pure python loop with integer arithmetic on bitmasks is likely faster than numpy overhead for such small arrays.
    
    # Let's stick to the bit-parallel idea if possible, or just simple loop.
    # Simple loop:
    # for mask in range(2**15):
    #   loads = np.zeros(10)
    #   for i in range(15):
    #     if mask & (1<<i):
    #       # add ccw load
    #     else:
    #       # add cw load
    
    # This is 32768 * 15 * (cost of adding 10 floats) ~ 5 million ops. Very fast.
    
    # Let's optimize by precomputing the load contribution of each demand in both directions.
    # load_cw[i] is an array of 10 floats, where load_cw[i][e] = w_i if e in cw_mask[i] else 0
    # load_ccw[i] is an array of 10 floats, where load_ccw[i][e] = w_i if e in ccw_mask[i] else 0
    
    load_cw = np.zeros((n_demands, n_edges))
    load_ccw = np.zeros((n_demands, n_edges))
    
    for i in range(n_demands):
        for e in range(n_edges):
            if (cw_masks[i] >> e) & 1:
                load_cw[i, e] = weights[i]
            else:
                load_ccw[i, e] = weights[i]

    # Now enumerate
    # We can vectorize the load calculation for a given routing vector r (0/1)
    # loads = sum_i r_i * load_ccw[i] + (1-r_i) * load_cw[i]
    # loads = sum_i load_cw[i] + sum_i r_i * (load_ccw[i] - load_cw[i])
    # base_load = sum_i load_cw[i]
    # delta_load = load_ccw[i] - load_cw[i]
    
    base_load = np.sum(load_cw, axis=0)
    delta_load = load_ccw - load_cw # shape (15, 10)
    
    # For a given routing vector r (0/1), total_load = base_load + r @ delta_load.T
    # We want to minimize max(total_load)
    
    # Enumerate all 2^15 combinations
    # 32768 iterations.
    # r @ delta_load.T is a (15,) dot (15,10) -> (10,) operation.
    # This is very fast in numpy.
    
    for mask in range(1 << n_demands):
        # Check time occasionally
        if time.time() > deadline - 5.0:
            break
            
        # Construct routing vector
        # We can use a lookup or just extract bits
        # r = np.array([(mask >> i) & 1 for i in range(n_demands)])
        # This list comprehension is slow.
        
        # Alternative: precompute all routing vectors? 32768 * 15 floats = 500k floats. Small.
        # But we can just compute the load incrementally?
        # Or just use the matrix multiplication approach with a precomputed matrix of all routings?
        # R is (32768, 15). Loads = R @ delta_load.T + base_load
        # This would be a (32768, 15) @ (15, 10) = (32768, 10) matrix.
        # Then max over axis=1 -> (32768,).
        # This is a single large matrix multiplication, which is highly optimized in BLAS.
        
        pass # We'll do this outside the loop
    
    # Let's do the vectorized approach for all masks at once.
    # It's memory efficient enough: 32768 * 15 * 8 bytes ~ 4 MB.
    
    # Generate all masks
    masks = np.arange(1 << n_demands, dtype=np.int32)
    
    # Convert masks to binary vectors
    # bits = (masks[:, None] >> np.arange(n_demands)) & 1
    bits = (masks[:, None] >> np.arange(n_demands)) & 1
    
    # Calculate loads for all combinations
    # loads = bits @ delta_load.T + base_load
    # bits: (32768, 15)
    # delta_load.T: (10, 15)
    # result: (32768, 10)
    
    loads = bits @ delta_load.T + base_load
    
    # Find max load for each combination
    max_loads = np.max(loads, axis=1)
    
    # Find the best combination
    best_idx = np.argmin(max_loads)
    best_max_load = max_loads[best_idx]
    
    # Extract the routing for the best combination
    best_mask = masks[best_idx]
    best_routing = (best_mask[:, None] >> np.arange(n_demands)) & 1 if False else ((best_mask >> np.arange(n_demands)) & 1).astype(float)
    
    # Ensure correct return type and shape
    return best_routing

# EVOLVE-BLOCK-END
