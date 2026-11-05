import numpy as np
import time

N = 400

def evaluate(h):
    # h is a numpy array
    conv = np.convolve(h, h)
    max_val = np.max(conv)
    sum_h = np.sum(h)
    if sum_h == 0:
        return float('inf')
    return abs(2 * N * max_val / (sum_h ** 2))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Start with seed
    h = np.ones(N)
    best_val = evaluate(h)
    best_h = h.copy()
    
    # Local search using small perturbations
    # Since the landscape is likely flat at the minimum (or it's a boundary),
    # we try to move towards regions with lower max correlation.
    # We scale h to keep sum=1 for stability during search?
    # No, value is scale invariant.
    
    # Try a few random directions
    for i in range(10000):
        if time.time() > deadline:
            break
        
        # Generate a random perturbation
        # Use a smooth perturbation to avoid high frequency noise
        pert = np.random.randn(N)
        # Low pass filter the perturbation
        pert = np.convolve(pert, np.ones(5)/5, mode='same')
        
        h_new = best_h + 0.1 * pert
        val = evaluate(h_new)
        
        if val < best_val:
            best_val = val
            best_h = h_new.copy()
            # If we found an improvement, try to refine
            for j in range(100):
                if time.time() > deadline:
                    break
                pert = np.random.randn(N)
                pert = np.convolve(pert, np.ones(5)/5, mode='same')
                h_try = best_h + 0.01 * pert
                val_try = evaluate(h_try)
                if val_try < best_val:
                    best_val = val_try
                    best_h = h_try.copy()

    return best_h
