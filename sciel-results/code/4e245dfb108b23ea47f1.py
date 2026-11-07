import numpy as np

N = 100

def compute_value(h):
    """Compute the objective value: max correlation / N * 2"""
    corr = np.correlate(h, 1.0 - h, mode="full")
    return np.max(corr) / N * 2

def local_search(h, max_iter=5000, time_limit=60.0):
    """Perform greedy local search by swapping pairs to reduce max correlation."""
    import time
    start_time = time.time()
    
    current_val = compute_value(h)
    ones = np.where(h > 0.5)[0]
    zeros = np.where(h < 0.5)[0]
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        best_swap = None
        best_val = current_val
        
        # Try all possible swaps
        for i in ones:
            for j in zeros:
                # Perform swap
                h[i], h[j] = h[j], h[i]
                
                # Compute new value
                new_val = compute_value(h)
                
                # Undo swap
                h[i], h[j] = h[j], h[i]
                
                if new_val < best_val:
                    best_val = new_val
                    best_swap = (i, j)
        
        if best_swap is None:
            break
        
        # Apply best swap
        i, j = best_swap
        h[i], h[j] = h[j], h[i]
        current_val = best_val
        
        # Update lists
        ones = np.where(h > 0.5)[0]
        zeros = np.where(h < 0.5)[0]
    
    return h, current_val

def construct():
    np.random.seed(42)
    
    best_h = None
    best_val = float('inf')
    
    # Try multiple initial configurations
    num_restarts = 20
    
    for restart in range(num_restarts):
        # Create a random binary configuration with 50 ones and 50 zeros
        h = np.zeros(N)
        ones_positions = np.random.choice(N, N//2, replace=False)
        h[ones_positions] = 1.0
        
        # Run local search
        h_optimized, val = local_search(h.copy(), max_iter=3000, time_limit=5.0)
        
        if val < best_val:
            best_val = val
            best_h = h_optimized.copy()
    
    return best_h
