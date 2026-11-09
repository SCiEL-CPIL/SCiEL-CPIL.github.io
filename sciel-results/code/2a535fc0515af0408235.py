def local_improve_vectorized(adj, labels, n, max_rounds=100):
    """Perform vectorized local search with simulated annealing."""
    # Convert to -1/1 representation
    x = 2.0 * labels - 1.0
    
    # Simulated Annealing parameters
    T_start = 5.0
    T_end = 0.1
    
    # We will do a fixed number of "passes".
    # Each pass consists of multiple iterations of flipping positive gain vertices.
    
    for r in range(max_rounds):
        T = T_start * (T_end / T_start) ** (r / max_rounds)
        
        # Inner loop: Flip all positive gain vertices until none remain
        # Limit inner iterations to prevent infinite loops in case of oscillation (though rare for LO)
        for _ in range(10):
            y = adj @ x
            gains = y * x
            
            # Flip vertices with positive gain
            pos_mask = gains > 0
            if not np.any(pos_mask):
                break
            
            # To avoid oscillation, we can flip only a fraction, or all.
            # Flipping all is standard LO.
            x[pos_mask] *= -1
            
        # Simulated Annealing step: Accept some negative moves to escape local minima
        # Compute current gains
        y = adj @ x
        gains = y * x
        
        neg_mask = gains < 0
        if np.any(neg_mask):
            # Accept with probability exp(gain/T)
            # To save time, only evaluate a random subset of negative gain vertices?
            # Or all? n=100 is small.
            neg_indices = np.where(neg_mask)[0]
            if len(neg_indices) > 0:
                gains_neg = gains[neg_indices]
                probs = np.exp(gains_neg / T)
                random_vals = np.random.rand(len(neg_indices))
                accept_mask = random_vals < probs
                
                if np.any(accept_mask):
                    flip_indices = neg_indices[accept_mask]
                    x[flip_indices] *= -1
        
    # Convert back to 0/1
    labels = (x > 0).astype(int)
    return labels
