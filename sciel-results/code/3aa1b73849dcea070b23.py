"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import differential_evolution, minimize

N = 600
STEP = 1.0 / N  # step size for intervals on [-1/4, 1/4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a sequence of 600 nonnegative step heights on equal intervals of [-1/4, 1/4]
    that minimizes the normalized maximum of the autoconvolution (upper bound on C1).
    
    Strategy:
    1. Parametrize the step function as a sum of K rectangular blocks.
    2. Each block is defined by center, width, and amplitude.
    3. Use differential evolution to search for optimal parameters.
    4. Refine with local optimization.
    5. Return the best discrete step function found.
    """
    t0 = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Baseline: flat function
    flat = np.ones(N)
    
    def autoconv_max_normalized(f):
        """Compute max of autoconvolution normalized by (sum f)^2 * N."""
        f = np.asarray(f, dtype=np.float64)
        if np.any(f < -1e-12):
            return np.inf
        f = np.maximum(f, 0.0)
        s = np.sum(f)
        if s < 1e-12:
            return np.inf
        # Autoconvolution via FFT
        nfft = 2 * N
        F = np.fft.rfft(f, n=nfft)
        ac = np.fft.irfft(F * F.conj(), n=nfft)[:N]
        max_ac = np.max(ac)
        return max_ac / (s * s) * N

    baseline_val = autoconv_max_normalized(flat)
    best_f = flat.copy()
    best_val = baseline_val
    
    # Parametric model: Sum of K rectangles
    # Parameters for each rectangle: [center, width, amplitude]
    # We will try K=1, K=2, K=3 rectangles.
    
    def param_to_step(params, K):
        """Convert parameters to a step function of length N."""
        f = np.zeros(N)
        for i in range(K):
            center = params[3*i]
            width = params[3*i+1]
            amp = params[3*i+2]
            
            # Clip width to be positive and within domain
            width = max(width, 1e-6)
            amp = max(amp, 0.0)
            
            # Convert continuous center/width to discrete indices
            # Domain is [-1/4, 1/4], mapped to [0, N-1]
            # center in [-0.25, 0.25] -> index in [0, N]
            c_idx = (center + 0.25) * N
            w_idx = width * N
            
            start_idx = int(np.floor(c_idx - w_idx/2.0))
            end_idx = int(np.ceil(c_idx + w_idx/2.0))
            
            # Clip to valid range
            start_idx = max(0, start_idx)
            end_idx = min(N, end_idx)
            
            if end_idx > start_idx:
                f[start_idx:end_idx] += amp
                
        return f

    def objective_param(params, K):
        """Objective function for the parametric model."""
        f = param_to_step(params, K)
        return autoconv_max_normalized(f)

    # Search spaces for K rectangles
    # Center: [-0.25, 0.25]
    # Width: [0.01, 0.5] (max width is half the domain length 0.5)
    # Amplitude: [0.1, 10.0]
    
    bounds = [(-0.25, 0.25), (0.01, 0.5), (0.1, 10.0)] * 3
    # For K=1: 3 params
    # For K=2: 6 params
    # For K=3: 9 params
    
    K_values = [1, 2, 3]
    
    for K in K_values:
        if time.time() - t0 > deadline:
            break
            
        bounds_K = bounds[:3*K]
        
        # Use differential evolution for global search
        # popsize=10, maxiter=100 is a reasonable start, adjust based on time
        # We want to be fast, so we limit iterations and popsize.
        
        try:
            res = differential_evolution(
                objective_param, 
                bounds_K, 
                args=(K,),
                seed=42,
                maxiter=50,
                popsize=10,
                tol=1e-4,
                polish=False # We'll do our own polish if needed, or skip for speed
            )
            
            if res.success or res.fun < best_val:
                best_params = res.x
                best_val_K = res.fun
                
                # Local refinement using L-BFGS-B
                # We need to handle the non-differentiability of the max and discrete step.
                # Since the parametric function is piecewise constant in f, the objective is piecewise smooth.
                # We can try a simple local search or just accept the DE result.
                # Let's try a few local refinements with Nelder-Mead or similar.
                
                from scipy.optimize import minimize as sp_minimize
                res2 = sp_minimize(
                    objective_param, 
                    best_params, 
                    args=(K,),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                
                if res2.fun < best_val_K:
                    best_val_K = res2.fun
                    best_params = res2.x
                
                if best_val_K < best_val:
                    best_val = best_val_K
                    best_f = param_to_step(best_params, K)
                    
        except Exception as e:
            pass # Continue to next K if error occurs

    # Check if we improved over baseline
    final_val = autoconv_max_normalized(best_f)
    if final_val < baseline_val:
        return list(best_f)
    
    # Return flat function if no improvement
    return list(flat)
# EVOLVE-BLOCK-END
