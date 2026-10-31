"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: 
1. Represent the step function as the squared magnitude of a low-order Fourier series 
   (cosine terms) to ensure non-negativity and smoothness.
2. Optimize the Fourier coefficients using Differential Evolution (global search) 
   with a soft-max surrogate objective.
3. Refine the solution using Nelder-Mead local optimization.
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 600
TIME_LIMIT = 140  # Safety margin before 150s kill

# Pre-compute grid points
# Interval [-1/4, 1/4], length 0.5
x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.5 / N
# Alternatively, standard midpoint rule:
x = -0.25 + (np.arange(N) + 0.5) * (0.5 / N)

# Define Fourier basis
# L = 0.5
L = 0.5
# Harmonics k=0, 1, 2, 3, 4
K = 4
# phi_k(x) = cos(2 * pi * k * x / L)
# k=0 -> cos(0) = 1
basis = np.zeros((K + 1, N))
for k in range(K + 1):
    if k == 0:
        basis[k] = 1.0
    else:
        basis[k] = np.cos(2 * np.pi * k * x / L)

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.fft.irfft(np.fft.rfft(f) * np.fft.rfft(f))[:N] # Full conv is 2N-1, but we only need max. 
    # Wait, np.convolve mode='full' is safer for exact max if peak is at edges? 
    # Autoconvolution of a function supported on [-1/4, 1/4] is supported on [-1/2, 1/2].
    # The peak is usually at 0 (center). 
    # Let's use full convolution to be safe.
    conv_full = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv_full) / (s * s)

def soft_max(conv, temperature=0.01):
    """Smooth approximation of max using log-sum-exp."""
    scaled = conv / temperature
    max_val = np.max(scaled)
    if max_val > 500:
        return max_val * temperature
    return temperature * (max_val + np.log(np.sum(np.exp(scaled - max_val))))

def compute_soft_score(f, temperature=0.01):
    """Compute smooth normalized max of autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    f = np.maximum(f, 0.0)
    conv_full = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return soft_max(conv_full, temperature) / (s * s)

def params_to_f(params):
    """Convert Fourier coefficients to step function heights."""
    # params: [c0, c1, c2, c3, c4]
    # f(x) = (sum ck * phi_k(x))^2
    linear_combo = np.dot(params, basis)
    f = linear_combo ** 2
    return f

def objective(params, temperature=0.01):
    """Objective function for optimization."""
    f = params_to_f(params)
    # Ensure non-negativity (squared is always non-negative, but numerical issues?)
    f = np.maximum(f, 0.0)
    return compute_soft_score(f, temperature)

def construct():
    start_time = time.time()
    
    # Initial best: Uniform (all 1s)
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Define bounds for Differential Evolution
    # c0 should be positive to keep function large enough
    # ck can be negative
    bounds = [(0.0, 2.0)] + [(-1.0, 1.0)] * K
    
    # Phase 1: Differential Evolution
    # Use soft-max with moderate temperature for smoothness
    temp_de = 0.05
    
    def de_objective(params):
        return objective(params, temp_de)
    
    try:
        # Differential Evolution
        result_de = differential_evolution(
            de_objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            tol=1e-5, 
            polish=False, 
            init='sobol',
            popsize=10
        )
        
        if time.time() - start_time < TIME_LIMIT - 20:
            best_params = result_de.x
            best_f = params_to_f(best_params)
            best_score = compute_score(best_f)
            
            # Phase 2: Local Refinement with Nelder-Mead
            # Use a sharper soft-max or exact max for refinement
            temp_nm = 0.01
            
            def nm_objective(params):
                return objective(params, temp_nm)
            
            result_nm = minimize(
                nm_objective, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if result_nm.success or result_nm.fun < best_score:
                if time.time() - start_time < TIME_LIMIT - 5:
                    best_params = result_nm.x
                    best_f = params_to_f(best_params)
                    best_score = compute_score(best_f)
                    
    except Exception:
        pass

    # Phase 3: Final Local Refinement on the full vector? 
    # The parametric family might be restrictive. 
    # Let's do a few steps of gradient descent on the full vector from the best parametric solution.
    
    if time.time() - start_time < TIME_LIMIT - 10:
        current_f = best_f.copy()
        current_score = best_score
        
        # Simple gradient descent on the full vector with simplex projection
        # This might break the parametric structure but can find better local minima.
        # However, the parametric family is a strong prior. 
        # Let's stick to the parametric refinement to avoid falling into bad local minima of the full space.
        
        # Try a few more temperatures in Nelder-Mead if time permits
        for temp in [0.005, 0.001]:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            def nm_objective_temp(params):
                return objective(params, temp)
            
            try:
                result_nm2 = minimize(
                    nm_objective_temp, 
                    best_params, 
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-12}
                )
                if result_nm2.fun < best_score:
                    best_params = result_nm2.x
                    best_f = params_to_f(best_params)
                    best_score = compute_score(best_f)
            except Exception:
                pass

    # Scale the function to have reasonable magnitudes
    # The score is scale-invariant, but returning very small or large numbers might cause issues.
    # Normalize so that max is 1.
    max_val = np.max(best_f)
    if max_val > 0:
        best_f = best_f / max_val
    else:
        best_f = np.ones(N) / N

    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
