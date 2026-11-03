import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import BSpline

N = 600
H = 0.5 / N
X = -0.25 + (np.arange(N) + 0.5) * H

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    def compute_score(heights):
        n_fft = 2048
        F = np.fft.rfft(heights, n=n_fft)
        conv = np.fft.irfft(F * F, n=n_fft) * H
        max_conv = np.max(conv)
        total_mass = np.sum(heights) * H
        return max_conv / (total_mass ** 2)

    # Use a set of B-spline basis functions
    # Knots at the edges and some internal points
    # Let's use a simple basis: shifted and scaled Gaussians or Cosines
    # Actually, let's just optimize a Super-Gaussian: (1 - (x/0.25)^2)^p
    # This is smooth and non-negative.
    
    def super_gaussian(p):
        # p > 0
        # f(x) = (1 - (x/0.25)^2)^p
        # This is 0 at x=0.25.
        return (1 - (X / 0.25)**2)**p

    best_val = float('inf')
    best_heights = None
    
    # Search over p
    ps = np.linspace(0.1, 10.0, 50)
    for p in ps:
        heights = super_gaussian(p)
        # Ensure non-negative
        heights = np.maximum(heights, 0)
        val = compute_score(heights)
        if val < best_val:
            best_val = val
            best_heights = heights.copy()
            
        if time.time() > deadline - 10:
            break

    # Local optimization using L-BFGS-B on the shape parameters
    # We can parameterize the function as a sum of a few Gaussians
    # Or just refine the Super-Gaussian with a local search on the array
    
    if best_heights is not None and time.time() < deadline - 20:
        current = best_heights.copy()
        current_val = best_val
        
        # Simple gradient-free local search
        np.random.seed(42)
        step_size = 0.01
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            # Random perturbation
            mask = np.random.random(N) < 0.1
            if mask.sum() == 0: continue
            pert = (np.random.random(mask.sum()) - 0.5) * step_size
            new_h = current.copy()
            new_h[mask] += pert
            new_h = np.maximum(new_h, 0)
            
            val = compute_score(new_h)
            if val < current_val:
                current = new_h
                current_val = val
                if current_val < best_val:
                    best_val = current_val
                    best_heights = current.copy()
            else:
                # Reduce step size if stuck?
                pass

    if best_heights is None:
        return [1.0] * N
    return list(best_heights)
