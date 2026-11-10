import numpy as np

def ring_routing() -> np.ndarray:
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
        (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Precompute load matrices for CW and CCW
    # M_CW[i, e] = weight of demand i if routed CW on edge e, else 0
    # M_CCW[i, e] = weight of demand i if routed CCW on edge e, else 0
    M_CW = np.zeros((n_demands, n_nodes))
    M_CCW = np.zeros((n_demands, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        if s < t:
            cw_edges = list(range(s, t))
        elif s > t:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))
        else:
            cw_edges = []
            
        cw_set = set(cw_edges)
        ccw_edges = [e for e in range(n_nodes) if e not in cw_set]
        
        for e in cw_edges:
            M_CW[i, e] = w
        for e in ccw_edges:
            M_CCW[i, e] = w
            
    # Base load if all routed CW
    L_base = np.sum(M_CW, axis=0)
    
    # Delta matrix: M_CCW - M_CW
    Delta_M = M_CCW - M_CW
    
    # Generate all 2^15 configurations
    # X is a matrix of shape (2^15, 15) where each row is a binary vector
    # Using bitwise operations for speed
    # np.arange(2**15) gives 0 to 32767
    # We need to expand these into 15 bits
    # Method: use np.unravel_index or bitwise shifts
    
    # Efficient way to create the configuration matrix
    # indices: 0 to 2^15 - 1
    indices = np.arange(1 << n_demands)
    
    # Create a matrix where column j contains the j-th bit of each index
    # bits = (indices[:, None] >> np.arange(n_demands)[None, :]) & 1
    # This creates a (32768, 15) matrix
    X = (indices[:, None] >> np.arange(n_demands)[None, :]) & 1
    
    # Compute loads for all configurations
    # Loads_all[i, e] = L_base[e] + sum_j X[i, j] * Delta_M[j, e]
    # This is equivalent to L_base + X @ Delta_M
    Loads_all = L_base[None, :] + X @ Delta_M
    
    # Find the configuration with the minimum maximum edge load
    max_loads = np.max(Loads_all, axis=1)
    best_idx = np.argmin(max_loads)
    
    # Extract the best routing configuration
    best_config = X[best_idx]
    
    # Return as float array: 0.0 for CW, 1.0 for CCW
    # The task says: "An entry >= 0.5 routes the demand counter-clockwise, any other value clockwise."
    return best_config.astype(float)
