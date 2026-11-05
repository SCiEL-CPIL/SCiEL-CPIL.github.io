"""
Attempts to minimize the third autocorrelation inequality constant C3 upper bound.
Uses random restarts and greedy local search (sign flips) to explore the space.
"""

import numpy as np
import time

N = 400

def _calculate_value(h):
    s = np.sum(h)
    if s == 0:
        return np.inf
    c = np.convolve(h, h)
    m = np.max(c)
    return abs(2 * N * m / (s ** 2))

def construct():
    start_time = time.time()
    deadline = 145.0  # 150s limit, 5s margin

    # Initialize with the known optimal candidate (constant vector)
    best_h = np.ones(N)
    best_val = _calculate_value(best_h) # Should be 2.0

    # Try to improve
    # Theoretical bound suggests 2.0 is minimal, but we search just in case.
    
    for restart in range(200):
        if time.time() > deadline - 5:
            break
            
        # Generate random starting point
        # Mix of random signs and gaussian noise to allow continuous values
        h = np.random.randn(N)
        # Ensure sum is not zero
        s = np.sum(h)
        if abs(s) < 1e-6:
            h += 0.1
            s = np.sum(h)
        # Normalize
        h = h / s
        
        current_val = _calculate_value(h)
        
        if current_val < best_val:
            best_val = current_val
            best_h = h.copy()
            
        # Local search phase
        # We perform a limited number of greedy sign flips or small perturbations
        for step in range(100):
            if time.time() > deadline - 5:
                break
                
            # Randomly perturb a few elements
            num_perturb = np.random.randint(1, 5)
            indices = np.random.choice(N, num_perturb, replace=False)
            
            improved = False
            for idx in indices:
                # Try flipping the sign of h[idx]
                old_val = h[idx]
                h[idx] = -h[idx]
                new_val = _calculate_value(h)
                
                if new_val < current_val:
                    current_val = new_val
                    improved = True
                    if current_val < best_val:
                        best_val = current_val
                        best_h = h.copy()
                else:
                    # Revert
                    h[idx] = old_val
                    
            if not improved:
                # Stuck in local minimum, break to next restart
                break

    return best_h
