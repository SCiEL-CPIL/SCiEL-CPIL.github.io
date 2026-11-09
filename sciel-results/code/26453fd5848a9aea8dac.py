import time
import numpy as np

def ring_routing() -> np.ndarray:
    """
    Solves the SONET ring loading problem by exhaustively searching all 2^15 
    routing configurations to find the one that minimizes the maximum edge load.
    
    Returns:
        np.ndarray of shape (15,) where 0.0 indicates clockwise and 1.0 indicates 
        counter-clockwise routing for each demand.
    """
    start_time = time.time()
    deadline = 600.0 # 600s limit, keep safety margin
    
    N = 10
    D = 15
    
    # Define demands: (source, target, weight)
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
    
    # Precompute edge usage for CW and CCW for each demand
    cw_masks = []
    ccw_masks = []
    
    for s, t, w in demands:
        cw_edges = np.zeros(N, dtype=bool)
        ccw_edges = np.zeros(N, dtype=bool)
        
        # Determine CW path edges
        if s <= t:
            cw_edges[s:t] = True
        else:
            cw_edges[s:] = True
            cw_edges[:t] = True
            
        # CCW uses the other edges
        ccw_edges = ~cw_edges
        
        cw_masks.append(cw_edges)
        ccw_masks.append(ccw_edges)
        
    cw_mask_matrix = np.array(cw_masks)    # shape (15, 10)
    ccw_mask_matrix = np.array(ccw_masks)  # shape (15, 10)
    weights = np.array([d[2] for d in demands], dtype=np.float64) # shape (15,)
    
    # Base load if all demands were routed CW
    base_load = np.sum(weights[:, np.newaxis] * cw_mask_matrix, axis=0) # shape (10,)
    
    # Delta load if a demand switches from CW to CCW
    delta_load = weights[:, np.newaxis] * (ccw_mask_matrix.astype(np.float64) - cw_mask_matrix.astype(np.float64)) # shape (15, 10)
    
    # Generate all 2^15 states
    num_states = 1 << D
    state_indices = np.arange(num_states, dtype=np.int32)
    
    # Create M matrix efficiently
    M = np.zeros((num_states, D), dtype=np.float64)
    for d in range(D):
        M[:, d] = ((state_indices >> d) & 1).astype(np.float64)
        
    # Calculate loads for all states
    loads = M @ delta_load + base_load[np.newaxis, :] # shape (32768, 10)
    
    # Find the state with minimum maximum load
    max_loads = np.max(loads, axis=1) # shape (32768,)
    best_idx = np.argmin(max_loads)
    
    # Retrieve the best routing
    best_routing = M[best_idx]
    
    return best_routing
