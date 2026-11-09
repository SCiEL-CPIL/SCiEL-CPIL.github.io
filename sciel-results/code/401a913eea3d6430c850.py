import numpy as np

def ring_routing() -> np.ndarray:
    # Demands: (source, target, weight)
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    
    # W_CW[i, e] is the weight of demand i on edge e if routed CW
    # W_CCW[i, e] is the weight of demand i on edge e if routed CCW
    W_CW = np.zeros((n_demands, n_nodes))
    W_CCW = np.zeros((n_demands, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        # Determine CW edges
        # Edges are e, e+1, ..., t-1 mod 10
        # If s < t, edges are s, ..., t-1
        # If s >= t, edges are s, ..., 9, 0, ..., t-1
        if s < t:
            cw_edges = list(range(s, t))
        else:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))
        
        for e in cw_edges:
            W_CW[i, e] = w
            
        # CCW edges are all other edges
        for e in range(n_nodes):
            if e not in cw_edges:
                W_CCW[i, e] = w
                
    # Base load if all demands are CW
    Base = np.sum(W_CW, axis=0) # shape (10,)
    
    # Difference matrix: Diff[i, e] = W_CCW[i, e] - W_CW[i, e]
    # If x_i = 1 (CCW), we add Diff[i, :] to the load
    # If x_i = 0 (CW), we add 0 (since Base already includes W_CW)
    # Load = Base + x @ Diff
    Diff = W_CCW - W_CW
    
    # Generate all 2^15 possible routing vectors
    # x is a matrix of shape (2^15, 15)
    # We can use np.meshgrid or bit manipulation
    # Simpler: use np.arange and binary representation
    max_val = 1 << n_demands
    # Create indices 0 to 2^15 - 1
    indices = np.arange(max_val)
    # Convert to binary, shape (max_val, 15)
    # We need to be careful with bit order. 
    # Bit i corresponds to demand i.
    # np.array2binary or manual bit extraction
    # Let's use a loop or vectorized bit extraction
    # powers = 2 ** np.arange(n_demands - 1, -1, -1) # MSB first
    # But we want demand 0 to be the least significant bit or most?
    # It doesn't matter as long as we are consistent.
    # Let's say bit 0 is demand 0.
    # x[k, i] = (indices[k] >> i) & 1
    
    # Vectorized bit extraction
    bits = np.array([np.array([(idx >> i) & 1 for i in range(n_demands)]) for idx in indices], dtype=float)
    # This loop might be slow for 32768 * 15. 
    # Better way:
    # powers = 2 ** np.arange(n_demands)
    # bits = (indices[:, None] // powers) % 2
    
    powers = 2 ** np.arange(n_demands)
    bits = (indices[:, None] // powers) % 2 # shape (32768, 15)
    
    # Compute loads for all routings
    # Loads = bits @ Diff.T + Base
    # bits is (32768, 15), Diff.T is (15, 10)
    # Result is (32768, 10)
    Loads = bits @ Diff.T + Base
    
    # Find max load for each routing
    MaxLoads = np.max(Loads, axis=1)
    
    # Find the routing with minimum max load
    best_idx = np.argmin(MaxLoads)
    best_routing = bits[best_idx]
    
    return best_routing
