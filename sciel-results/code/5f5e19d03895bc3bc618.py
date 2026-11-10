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
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, keep safety margin

    # Problem Definition
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
    n_edges = n_nodes
    
    # Precomputation
    # M[d, e] is 1 if demand d uses edge e when routed clockwise.
    # Clockwise from s to t uses edges s, s+1, ..., t-1 (mod n).
    M = np.zeros((D, n_edges), dtype=np.float64)
    weights = np.zeros(D, dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        weights[i] = w
        # Determine edges used clockwise
        # If s < t, edges are s, s+1, ..., t-1
        # If s > t, edges are s, s+1, ..., n-1, 0, ..., t-1
        # If s == t, no edges? But ring routing usually implies distinct nodes or full circle?
        # "Clockwise from s to t uses edges s, ..., t-1". If s==t, it's 0 edges? 
        # Standard ring routing: if s==t, load is 0. 
        # Let's check demands: no s==t in the list.
        
        if s < t:
            for e in range(s, t):
                M[i, e] = 1.0
        elif s > t:
            for e in range(s, n_nodes):
                M[i, e] = 1.0
            for e in range(0, t):
                M[i, e] = 1.0
        else:
            # s == t, no edges used in either direction? 
            # Usually this implies the demand is local, load 0.
            pass
            
    # We want to minimize max_e (Base_e + sum_d x_d * Delta_d_e)
    # Load_e = sum_d w_d * I(d uses e)
    # I(d uses e) = M[d,e] if x_d=0 (CW), else 1-M[d,e] if x_d=1 (CCW)
    # I(d uses e) = M[d,e] * (1-x_d) + (1-M[d,e]) * x_d
    #             = M[d,e] + x_d * (1 - 2*M[d,e])
    
    # Base load if all CW (x_d=0)
    Base = M.T @ weights  # Shape (n_edges,)
    
    # Delta contribution per demand per edge
    # Delta[d, e] = w_d * (1 - 2*M[d,e])
    Delta = weights[:, np.newaxis] * (1.0 - 2.0 * M)  # Shape (D, n_edges)
    
    # We iterate over all 2^D combinations
    # x is a vector of 0s and 1s of length D.
    # Load = Base + Delta.T @ x
    
    best_max_load = float('inf')
    best_mask = 0
    total_combinations = 1 << D
    
    # Vectorized inner loop for speed?
    # Delta is (D, n_edges). x is (D,).
    # Load = Base + Delta.T @ x
    
    # We can pre-compute Delta.T as (n_edges, D)
    Delta_T = Delta.T
    
    for mask in range(total_combinations):
        # Check deadline
        if time.time() > deadline - 5.0:
            break
            
        # Convert mask to vector x
        # x_d = (mask >> d) & 1
        x = np.array([(mask >> d) & 1 for d in range(D)], dtype=np.float64)
        
        # Calculate loads
        loads = Base + Delta_T @ x
        
        # Find max load
        current_max_load = np.max(loads)
        
        if current_max_load < best_max_load:
            best_max_load = current_max_load
            best_mask = mask
            
    # Convert best_mask to routing array
    routing = np.array([(best_mask >> d) & 1 for d in range(D)], dtype=np.float64)
    
    return routing

# EVOLVE-BLOCK-END
