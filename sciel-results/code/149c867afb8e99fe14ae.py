"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    
    rng = np.random.RandomState(42)
    h = rng.choice([-1.0, 1.0], size=N)
    
    def get_cost(h):
        # Compute aperiodic autocorrelation
        c = np.correlate(h, h, mode='full')
        # The autoconvolution upper bound on C3 relates to the sidelobe energy
        # Exclude the main peak at index N-1 (lag 0)
        # Sum of squares of all autocorrelation coefficients
        return np.sum(c**2)
    
    current_cost = get_cost(h)
    best_cost = current_cost
    best_h = h.copy()
    
    # Local search with simulated annealing-like acceptance
    temp = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        i = rng.randint(N)
        old_val = h[i]
        h[i] = -old_val
        
        new_cost = get_cost(h)
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or (temp > 0.01 and rng.random() < np.exp(-delta_cost / (temp + 1e-10))):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_h = h.copy()
        else:
            h[i] = old_val
        
        temp *= cooling
    
    return best_h
# EVOLVE-BLOCK-END
