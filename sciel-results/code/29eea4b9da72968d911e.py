import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 20s safety margin
    
    # Initialize with the known optimal candidate (uniform)
    h = np.full(N, 0.5)
    
    # Define the objective function
    # Assuming cyclic shift by 1
    # Overlap is max of h[i] * (1 - h[(i+1)%N])
    def objective(x):
        # x is array of length N
        # Calculate h * (1 - h_shifted)
        h_shifted = np.roll(x, -1)
        overlaps = x * (1.0 - h_shifted)
        return np.max(overlaps)
    
    best_score = objective(h)
    best_h = h.copy()
    
    # Simple local search with gradient-free steps
    # We try to perturb h and keep if it improves
    # Since 0.5 is likely the global min, this might not improve, 
    # but it ensures we are robust to any slight metric variations.
    
    step_size = 0.1
    iters = 0
    max_iters = 100000
    
    while time.time() < deadline and iters < max_iters:
        # Generate a random perturbation
        # To maintain sum constraint, we can pick two indices i, j
        # and move delta from i to j
        i = np.random.randint(N)
        j = np.random.randint(N)
        if i == j:
            continue
        
        # Random delta
        delta = np.random.uniform(-step_size, step_size)
        
        # Ensure bounds [0, 1]
        if h[i] + delta < 0 or h[i] + delta > 1:
            continue
        if h[j] - delta < 0 or h[j] - delta > 1:
            continue
            
        h[i] += delta
        h[j] -= delta
        
        new_score = objective(h)
        
        if new_score < best_score:
            best_score = new_score
            best_h = h.copy()
            # Maybe decrease step size slowly?
            # step_size *= 0.99
        else:
            # Revert
            h[i] -= delta
            h[j] += delta
            
        iters += 1
        
    return best_h
