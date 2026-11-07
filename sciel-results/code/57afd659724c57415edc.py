import numpy as np

N = 100

def construct():
    # Initialize with 0.5
    h = np.full(N, 0.5)
    
    # Simulated Annealing
    # Objective: minimize max(corr)
    # We approximate max by sum of squares to make it smooth, 
    # but we actually evaluate max for acceptance.
    
    def get_value(h):
        g = 1.0 - h
        c = np.correlate(h, g, mode="full")
        return np.max(c) / N * 2.0

    current_val = get_value(h)
    
    # Annealing parameters
    T = 0.1
    T_min = 1e-6
    alpha = 0.99
    
    # Number of steps
    steps = 20000
    
    for i in range(steps):
        # Pick two indices to swap mass
        idx1 = np.random.randint(N)
        idx2 = np.random.randint(N)
        if idx1 == idx2:
            continue
            
        # Move small amount of mass
        # Ensure we stay in [0,1]
        # h[idx1] + delta, h[idx2] - delta
        max_delta = min(h[idx1], 1.0 - h[idx2])
        if max_delta <= 1e-9:
            continue
            
        delta = np.random.uniform(-1, 1) * min(0.1, max_delta)
        
        h_new = h.copy()
        h_new[idx1] += delta
        h_new[idx2] -= delta
        
        # Check bounds (should be fine with max_delta logic, but clip just in case)
        if h_new[idx1] < 0 or h_new[idx1] > 1 or h_new[idx2] < 0 or h_new[idx2] > 1:
            continue
            
        new_val = get_value(h_new)
        
        if new_val < current_val:
            h = h_new
            current_val = new_val
        else:
            # Accept worse solution with probability
            if np.random.rand() < np.exp(-(new_val - current_val) / T):
                h = h_new
                current_val = new_val
                
        T *= alpha
        if T < T_min:
            T = T_min

    # Final check
    if not np.isfinite(current_val):
        return np.full(N, 0.5)
        
    return h

# EVOLVE-BLOCK-END
