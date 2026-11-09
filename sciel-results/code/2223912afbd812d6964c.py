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
        1 -> route the demand counter-clockwise (since >= 0.5 is CCW)

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    start_time = time.time()
    deadline = 600  # seconds
    
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
        (5, 2, 2)
    ]
    D = len(demands)
    
    # Pre-compute masks
    # cw_mask[d, e] is 1 if edge e is used when demand d is routed clockwise
    # ccw_mask[d, e] is 1 if edge e is used when demand d is routed counter-clockwise
    cw_mask = np.zeros((D, n_nodes), dtype=np.int8)
    ccw_mask = np.zeros((D, n_nodes), dtype=np.int8)
    weights = np.zeros(D, dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        weights[i] = w
        # Clockwise path: s -> s+1 -> ... -> t
        # Edges are (s, s+1), ..., (t-1, t)
        # Edge index e corresponds to edge (e, (e+1)%n)
        # So clockwise uses edges s, s+1, ..., t-1 (mod n)
        # If s < t: edges s, s+1, ..., t-1
        # If s > t: edges s, s+1, ..., n-1, 0, ..., t-1
        # If s == t: no edges (but demands are distinct nodes usually, here s!=t)
        
        # Counter-clockwise path: s -> s-1 -> ... -> t
        # Edges are (s, s-1), ..., (t+1, t)
        # Edge index e corresponds to edge (e, (e+1)%n).
        # The edge connecting u and v (where u = v-1 mod n) is indexed by u.
        # So CCW from s to t uses edges: s-1, s-2, ..., t (mod n)
        # Wait. 
        # Edge e joins e and (e+1)%n.
        # CW from s to t:
        #   Move from s to s+1 uses edge s.
        #   Move from s+1 to s+2 uses edge s+1.
        #   ...
        #   Move from t-1 to t uses edge t-1.
        #   So edges are s, s+1, ..., t-1 (mod n).
        
        # CCW from s to t:
        #   Move from s to s-1 uses edge s-1 (since edge s-1 joins s-1 and s).
        #   Move from s-1 to s-2 uses edge s-2.
        #   ...
        #   Move from t+1 to t uses edge t.
        #   So edges are s-1, s-2, ..., t (mod n).
        
        # Let's verify with an example.
        # n=10. s=0, t=1.
        # CW: 0->1. Edge 0. Formula: 0 to 0 (inclusive). Correct.
        # CCW: 0->9->8...->1. Edges: 9, 8, 7, 6, 5, 4, 3, 2, 1.
        # Formula: s-1=9 down to t=1. Correct.
        
        # s=0, t=5.
        # CW: 0->1->2->3->4->5. Edges 0,1,2,3,4.
        # Formula: 0 to 4. Correct.
        # CCW: 0->9->8->7->6->5. Edges 9,8,7,6.
        # Formula: s-1=9 down to t=5. Correct.
        
        # s=5, t=0.
        # CW: 5->6->7->8->9->0. Edges 5,6,7,8,9.
        # Formula: 5 to 9. Correct.
        # CCW: 5->4->3->2->1->0. Edges 4,3,2,1,0.
        # Formula: s-1=4 down to t=0. Correct.
        
        # Implementation of range:
        if s < t:
            cw_edges = range(s, t)
            ccw_edges = range(s - 1, t - 1, -1) # s-1 down to t
            # Wait, range(s-1, t-1, -1) stops at t. So it includes t.
            # Example s=0, t=1. range(-1, 0, -1) -> [-1] which is 9. Correct.
            # Example s=0, t=5. range(-1, 4, -1) -> [9, 8, 7, 6, 5]. Correct.
            # Example s=5, t=0. s < t is false.
        else:
            # s >= t
            if s == t:
                cw_edges = []
                ccw_edges = []
            else:
                cw_edges = list(range(s, n_nodes)) + list(range(0, t))
                ccw_edges = range(s - 1, t - 1, -1) # s-1 down to t
                # Example s=5, t=0. range(4, -1, -1) -> [4, 3, 2, 1, 0]. Correct.
                # Example s=6, t=0. range(5, -1, -1) -> [5, 4, 3, 2, 1, 0]. Correct.
                # Example s=6, t=2. range(5, 1, -1) -> [5, 4, 3, 2]. Correct.
        
        for e in cw_edges:
            cw_mask[i, e % n_nodes] = 1
            
        for e in ccw_edges:
            ccw_mask[i, e % n_nodes] = 1

    # Brute force all 2^15 combinations
    best_max_load = float('inf')
    best_mask = 0
    
    # Pre-compute weight-weighted masks?
    # load = weights @ ((1-b)*cw_mask + b*ccw_mask)
    # This is equivalent to:
    # load = weights @ cw_mask + weights @ (b * (ccw_mask - cw_mask))
    # Let base_cw_loads = weights @ cw_mask (shape 10)
    # Let delta_mask = ccw_mask - cw_mask (shape 15, 10)
    # Let delta_loads = weights @ delta_mask (shape 15, 10) -- wait, weights is (15,), delta_mask is (15,10).
    # weights @ delta_mask is (10,). This gives the change in load for each edge if ALL demands switched to CCW.
    # But we switch individually.
    
    # Let's just use the direct formula inside the loop, it's fast enough.
    # Or pre-compute:
    # L(b) = sum_d w_d * ( (1-b_d) * cw_mask[d] + b_d * ccw_mask[d] )
    # L(b) = sum_d w_d * cw_mask[d] + sum_d w_d * b_d * (ccw_mask[d] - cw_mask[d])
    # L(b) = L_cw_all + sum_d b_d * w_d * (ccw_mask[d] - cw_mask[d])
    
    L_cw_all = np.sum(weights[:, np.newaxis] * cw_mask, axis=0) # shape (10,)
    delta_per_demand = weights[:, np.newaxis] * (ccw_mask - cw_mask) # shape (15, 10)
    
    for mask in range(1 << D):
        if time.time() > deadline - 10:
            break
            
        # Convert mask to binary vector b
        # b_d is 1 if bit d is set in mask
        b = np.array([(mask >> d) & 1 for d in range(D)], dtype=np.float64)
        
        # Compute loads
        # L = L_cw_all + b @ delta_per_demand
        loads = L_cw_all + b @ delta_per_demand
        
        current_max_load = np.max(loads)
        
        if current_max_load < best_max_load:
            best_max_load = current_max_load
            best_mask = mask
            # Early termination if we find a perfect balance? 
            # Theoretical lower bound is max(sum of weights on any cut / 2)? 
            # Sum of weights = 3+2+4+1+5+2+3+2+4+1+3+2+5+1+2 = 40.
            # Total load on all edges if all CW is sum(weights)*avg_path_len? No.
            # Just keep searching.
            
    # Construct result
    result = np.array([(best_mask >> d) & 1 for d in range(D)], dtype=np.float64)
    
    return result

# EVOLVE-BLOCK-END
