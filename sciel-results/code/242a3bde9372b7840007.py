"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Use a smooth parametric family (e.g., Gaussian-like or cosine-shaped)
that naturally has low autoconvolution peaks, then refine with local optimization.

Key insight: The constant function [1]*600 gives a triangular autoconvolution
with peak at center. To reduce the normalized maximum, we want a function that
spreads mass more evenly or has a shape whose autoconvolution is flatter.

A cosine-shaped function: f(x) = cos(pi * x / (1/2)) for |x| <= 1/4, 0 otherwise
has a known autoconvolution that is smoother. But we need to work on discrete intervals.

Better approach: Start with a smooth parametric family and use Nelder-Mead or
differential evolution to optimize parameters that minimize the max of the
autoconvolution (normalized by the L1 norm or similar).

The normalized max is likely: max(f * f) / (sum(f) * sum(f) / N) or similar.
Let me think about what "normalized maximum of its autoconvolution" means.

Autoconvolution: (f * f)[k] = sum_i f[i] * f[k-i]
For a step function on 600 intervals, this is a discrete convolution.

"Normalized maximum" - probably max(conv) / (sum(f))^2 * N or max(conv) / sum(conv).
Since sum(conv) = (sum(f))^2, normalizing by sum(conv) gives max(conv) / (sum(f))^2.

Let's use: score = max(np.convolve(f, f, mode='full')) / (np.sum(f) ** 2)

We want to minimize this. The constant function gives a specific value.

Let me try a different structural approach: use a smooth basis (like a few Gaussians
or a polynomial) to represent the 600 values, then optimize the continuous parameters.
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

def gaussian_basis(x, centers, sigmas, heights):
    """Generate a smooth nonnegative function from Gaussian components."""
    result = np.zeros_like(x)
    for c, s, h in zip(centers, sigmas, heights):
        result += h * np.exp(-0.5 * ((x - c) / s) ** 2)
    return result

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
    
    f = gaussian_basis(x, centers, sigmas, heights)
    # Clip to nonnegative
    f = np.maximum(f, 0.0)
    return f

def objective(params, n_gaussians):
    f = parametric_to_step(params, n_gaussians)
    return compute_score(f)

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
        
        for restart in range(5):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            # Initialize parameters
            centers = rng.uniform(-0.2, 0.2, size=n_g)
            sigmas = rng.uniform(0.05, 0.15, size=n_g)
            heights = rng.uniform(0.5, 2.0, size=n_g)
            
            x0 = np.concatenate([centers, sigmas, heights])
            
            # Use Nelder-Mead optimization
            try:
                result = minimize(
                    objective, x0, args=(n_g,),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_params = result.x
            except Exception:
                pass
        
        if best_params is not None and best_obj < best_score:
            f = parametric_to_step(best_params, n_g)
            best_f = f
            best_score = best_obj
    
    # Local refinement: direct optimization on the 600 values
    if time.time() - start_time < TIME_LIMIT - 30:
        # Use a few iterations of local search
        current_f = best_f.copy()
        current_score = best_score
        
        for iteration in range(20):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            # Random perturbation
            perturbation = np.random.randn(N) * 0.01
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
