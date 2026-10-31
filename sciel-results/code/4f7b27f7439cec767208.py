"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Use a smooth parametric family based on cosine bumps with different
shapes, then refine with local optimization using a smooth max surrogate.

The key insight is to use a family of functions whose autoconvolution has a
flatter peak. Cosine bumps with varying widths and positions can create
functions with low autoconvolution peaks.

We use a smooth max (log-sum-exp) surrogate to make the objective
differentiable, allowing gradient-based optimization to work better.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 600
TIME_LIMIT = 130  # safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def smooth_max(x, beta):
    """Smooth approximation to max using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    # Scale to avoid overflow
    x_scaled = x / x_max
    shifted = x_scaled - np.max(x_scaled)
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max * (1.0 + log_sum)

def cosine_basis(x, centers, widths, heights):
    """Generate a smooth nonnegative function from cosine bump components.
    
    Each component is a cosine bump: h * cos(pi * (x-c)/w)^2 for |x-c| <= w/2, 0 otherwise
    """
    result = np.zeros_like(x)
    for c, w, h in zip(centers, widths, heights):
        dist = np.abs(x - c)
        mask = dist <= w / 2.0
        if np.any(mask):
            result[mask] += h * np.cos(np.pi * (x[mask] - c) / w) ** 2
    return result

def parametric_to_step(params, n_components):
    """Convert parameters to a step function of length N."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N  # interval centers
    centers = params[:n_components]
    widths = params[n_components:2*n_components]
    heights = params[2*n_components:3*n_components]
    
    # Ensure widths are positive and reasonable
    widths = np.abs(widths) + 0.01
    widths = np.minimum(widths, 0.5)  # max width is full domain
    # Ensure heights are nonnegative
    heights = np.abs(heights)
    
    f = cosine_basis(x, centers, widths, heights)
    # Clip to nonnegative
    f = np.maximum(f, 0.0)
    return f

def objective(params, n_components, beta=10.0):
    f = parametric_to_step(params, n_components)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    # Use smooth max for differentiability
    return smooth_max(conv, beta) / (s * s)

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different numbers of cosine bump components
    for n_c in [1, 2, 3, 4]:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        # Random initial parameters
        rng = np.random.RandomState(42)
        best_params = None
        best_obj = np.inf
        
        for restart in range(8):
            if time.time() - start_time > TIME_LIMIT - 15:
                break
            
            # Initialize parameters
            centers = rng.uniform(-0.2, 0.2, size=n_c)
            widths = rng.uniform(0.05, 0.2, size=n_c)
            heights = rng.uniform(0.5, 2.0, size=n_c)
            
            x0 = np.concatenate([centers, widths, heights])
            
            # Use Nelder-Mead optimization with smooth max surrogate
            try:
                result = minimize(
                    objective, x0, args=(n_c, 10.0),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9}
                )
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_params = result.x
            except Exception:
                pass
        
        if best_params is not None:
            f = parametric_to_step(best_params, n_c)
            actual_score = compute_score(f)
            if actual_score < best_score:
                best_f = f
                best_score = actual_score
    
    # Local refinement: direct optimization on the 600 values
    if time.time() - start_time < TIME_LIMIT - 20:
        # Use coordinate descent with smooth max
        current_f = best_f.copy()
        current_score = best_score
        
        for iteration in range(50):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            # Try small perturbations in random directions
            for _ in range(5):
                idx = np.random.randint(N)
                step = np.random.randn() * 0.005 * current_f[idx]
                candidate = current_f.copy()
                candidate[idx] = max(0.0, current_f[idx] + step)
                cand_score = compute_score(candidate)
                
                if cand_score < current_score:
                    current_f = candidate
                    current_score = cand_score
        
        if current_score < best_score:
            best_f = current_f
            best_score = current_score
    
    # Return as list of floats
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
