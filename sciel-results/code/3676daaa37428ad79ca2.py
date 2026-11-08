"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    def compute_c3(h):
        """Compute C3: circular autocorrelation at lag 3, normalized."""
        # h is array of ±1 of length N
        shifted = np.roll(h, -3)
        return np.dot(h, shifted) / N
    
    # Initialize with random ±1
    h = np.random.choice([-1, 1], size=N)
    best_h = h.copy()
    best_c3 = compute_c3(h)
    
    # Local search with multi-flip moves
    max_iterations = 100000
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Try flipping 2-3 random positions
        num_flips = np.random.choice([2, 3], p=[0.7, 0.3])
        indices = np.random.choice(N, size=num_flips, replace=False)
        
        h_new = h.copy()
        h_new[indices] *= -1
        c3_new = compute_c3(h_new)
        
        if c3_new < best_c3:
            best_c3 = c3_new
            best_h = h_new.copy()
            h = h_new
        else:
            # Revert
            h[indices] *= -1
    
    return best_h
# EVOLVE-BLOCK-END
