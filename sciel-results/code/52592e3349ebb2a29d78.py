import numpy as np
from scipy.optimize import differential_evolution
import time

N = 600
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution."""
    heights = np.asarray(heights)
    # Autoconvolution via FFT
    n_fft = 2 * N
    f_hat = np.fft.rfft(heights, n=n_fft)
    conv = np.fft.irfft(f_hat * f_hat, n=n_fft)
    # The autoconvolution of the step function
    # Normalize: max(conv) / (sum(heights))**2
    total = np.sum(heights)
    if total == 0:
        return float('inf')
    max_conv = np.max(conv)
    return max_conv / (total * total)

def construct():
    global START_TIME
    start = time.time()
    
    # Initial guess: uniform
    best_heights = np.ones(N)
    best_score = compute_score(best_heights)
    
    # Try a few analytical candidates first
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N))
    
    # 2. Edge-weighted (higher at edges)
    x = np.linspace(-1, 1, N)
    candidates.append(1.0 + 0.5 * x**2)
    
    # 3. Center-weighted (higher at center)
    candidates.append(1.0 - 0.5 * x**2)
    
    # 4. Cosine bump
    candidates.append(np.cos(np.pi * x / 2) ** 2)
    
    # 5. Two-lobe
    candidates.append(np.maximum(0, np.cos(2 * np.pi * x)))
    
    for cand in candidates:
        cand = np.maximum(0, cand)
        s = compute_score(cand)
        if s < best_score:
            best_score = s
            best_heights = cand.copy()
    
    # Check time
    if time.time() - start > TIME_LIMIT - 30:
        return best_heights.tolist()
    
    # Differential evolution on a reduced set of parameters
    # Use a parameterization: heights = base + sum of a few cosine modes
    # This reduces the search space from 600 to ~10-20 parameters
    
    def param_to_heights(params):
        """Convert parameters to nonnegative step heights."""
        # params[0] = base level
        # params[1:] = amplitudes of cosine modes
        base = params[0]
        heights = np.full(N, base)
        for i, amp in enumerate(params[1:]):
            mode = i + 1
            heights += amp * np.cos(2 * np.pi * mode * np.arange(N) / N)
        # Ensure nonnegative
        heights = np.maximum(0, heights)
        return heights
    
    def objective(params):
        heights = param_to_heights(params)
        return compute_score(heights)
    
    # Try several different mode counts
    for n_modes in [3, 5, 8]:
        if time.time() - start > TIME_LIMIT - 20:
            break
        
        n_params = 1 + n_modes
        bounds = [(0.1, 3.0)] + [(-2.0, 2.0)] * n_modes
        
        try:
            result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_heights = param_to_heights(result.x)
        except Exception:
            pass
    
    # Local refinement with Nelder-Mead on the full 600-dim problem
    if time.time() - start < TIME_LIMIT - 10:
        from scipy.optimize import minimize
        
        def full_objective(heights):
            heights = np.maximum(0, heights)
            return compute_score(heights)
        
        try:
            result = minimize(
                full_objective,
                best_heights,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_heights = np.maximum(0, result.x)
        except Exception:
            pass
    
    return best_heights.tolist()
