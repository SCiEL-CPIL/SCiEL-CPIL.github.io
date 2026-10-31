"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Use a smooth parametric family based on cosine bumps with different
shapes, then refine with local optimization using a smooth max surrogate.

The key insight is to use a family of functions whose autoconvolution has a
flatter peak. Cosine bumps with varying widths and positions can create
functions with low autoconvolution peaks.
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

def smooth_max(x, beta=100.0):
    """Soft-max surrogate using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    m = np.max(x)
    if m > 1e10:
        return m
    return m + np.log(np.sum(np.exp((x - m) / beta))) / beta

def parametric_to_step(params, n_gaussians):
    """Convert parameters to a step function of length N."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N  # interval centers
    centers = params[:n_gaussians]
    sigmas = params[n_gaussians:2*n_gaussians]
    heights = params[2*n_gaussians:3*n_gaussians]
    
    # Ensure sigmas are positive
    sigmas = np.abs(sigmas) + 1e-6
    # Ensure heights are nonnegative
    heights = np.abs(heights)
    
    # Build the function
    f = np.zeros(N)
    for c, s, h in zip(centers, sigmas, heights):
        f += h * np.exp(-0.5 * ((x - c) / s) ** 2)
    
    # Clip to nonnegative
    f = np.maximum(f, 0.0)
    return f

def objective(params, n_gaussians, beta=50.0):
    f = parametric_to_step(params, n_gaussians)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    # Use smooth max for better gradient behavior
    return smooth_max(conv, beta) / (s * s)

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different numbers of Gaussian components
    for n_g in [1, 2, 3]:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        
        # Random initial parameters
        rng = np.random.RandomState(42)
        best_params = None
        best_obj = np.inf
        
        for restart in range(8):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            # Initialize parameters
            centers = rng.uniform(-0.2, 0.2, size=n_g)
            sigmas = rng.uniform(0.05, 0.15, size=n_g)
            heights = rng.uniform(0.5, 2.0, size=n_g)
            
            x0 = np.concatenate([centers, sigmas, heights])
            
            # Use Nelder-Mead optimization with smooth objective
            try:
                result = minimize(
                    objective, x0, args=(n_g, 50.0),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9}
                )
                # Evaluate with exact objective
                f_candidate = parametric_to_step(result.x, n_g)
                exact_score = compute_score(f_candidate)
                if exact_score < best_obj:
                    best_obj = exact_score
                    best_params = result.x.copy()
            except Exception:
                pass
        
        if best_params is not None and best_obj < best_score:
            f = parametric_to_step(best_params, n_g)
            best_f = f
            best_score = best_obj
    
    # Local refinement: direct optimization on the 600 values
    if time.time() - start_time < TIME_LIMIT - 30:
        # Use a few iterations of local search with smaller perturbations
        current_f = best_f.copy()
        current_score = best_score
        
        for iteration in range(50):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            # Random perturbation with decreasing amplitude
            amp = 0.01 * (0.95 ** iteration)
            perturbation = np.random.randn(N) * amp
            candidate = np.maximum(current_f + perturbation, 0)
            cand_score = compute_score(candidate)
            
            if cand_score < current_score:
                current_f = candidate
                current_score = cand_score
        
        if current_score < best_score:
            best_f = current_f
            best_score = current_score
    
    # Final high-precision polish: try a few more refined local steps
    if time.time() - start_time < TIME_LIMIT - 10:
        current_f = best_f.copy()
        current_score = best_score
        
        for iteration in range(30):
            if time.time() - start_time > TIME_LIMIT - 2:
                break
            
            # Very small perturbations for fine-tuning
            amp = 0.003 * (0.9 ** iteration)
            perturbation = np.random.randn(N) * amp
            candidate = np.maximum(current_f + perturbation, 0)
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
