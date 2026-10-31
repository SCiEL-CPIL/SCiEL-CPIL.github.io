import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 600
L = 0.5  # interval length [-1/4, 1/4]

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution.
    
    For a step function with heights h_i on N equal intervals of total length L,
    the autoconvolution at lag 0 is sum(h_i^2 * dx) where dx = L/N.
    The normalization is by max(h) * sum(h_i) * dx.
    
    The ratio is (sum h_i^2) / (max(h) * sum h_i), which is scale invariant.
    """
    h = np.asarray(h)
    s1 = np.sum(h)
    s2 = np.sum(h**2)
    hmax = np.max(h)
    if s1 < 1e-15 or hmax < 1e-15:
        return np.inf
    return s2 / (hmax * s1)

def _gaussian_mixture_to_heights(params, n_gaussians):
    """Convert Gaussian mixture parameters to step function heights.
    
    params: array of shape [3 * n_gaussians] containing [amp1, mean1, std1, amp2, mean2, std2, ...]
    Returns: array of N nonnegative heights.
    """
    x = np.linspace(-0.25, 0.25, N)
    h = np.zeros(N)
    for i in range(n_gaussians):
        amp = params[3*i]
        mean = params[3*i+1]
        std = params[3*i+2]
        if std < 1e-8:
            std = 1e-8
        h += amp * np.exp(-0.5 * ((x - mean) / std)**2)
    return np.maximum(h, 0.0)

def _objective(params, n_gaussians):
    """Objective function for optimization."""
    h = _gaussian_mixture_to_heights(params, n_gaussians)
    val = compute_autoconv_max(h)
    if not np.isfinite(val):
        return 1e10
    return val

def construct():
    """Construct a nonnegative step function with 600 intervals on [-1/4, 1/4]
    that minimizes the normalized maximum of its autoconvolution."""
    
    deadline = time.time() + 140.0  # safety margin before 150s limit
    
    # Start with analytical spike solutions as initial best
    best_h = np.ones(N)
    best_val = compute_autoconv_max(best_h)
    
    # Single spike
    eps_opt = (np.sqrt(N) - 1) / (N - 1)
    h_spike = np.full(N, eps_opt)
    h_spike[0] = 1.0
    val_spike = compute_autoconv_max(h_spike)
    if val_spike < best_val:
        best_val = val_spike
        best_h = h_spike.copy()
    
    # Multiple spikes
    for k in [2, 3, 5, 10, 20, 50]:
        if k >= N:
            break
        if N > k:
            eps_k = (-k + np.sqrt(k * N)) / (N - k)
            if eps_k > 0 and eps_k < 1:
                h_k = np.full(N, eps_k)
                for i in range(k):
                    h_k[i] = 1.0
                val_k = compute_autoconv_max(h_k)
                if val_k < best_val:
                    best_val = val_k
                    best_h = h_k.copy()
    
    # Use Gaussian mixture representation with differential evolution
    # Try different numbers of Gaussian components
    for n_gaussians in [4, 6, 8, 10]:
        if time.time() > deadline - 15:
            break
        
        # Define bounds for each parameter
        # amp: [0, 2], mean: [-0.25, 0.25], std: [0.01, 0.25]
        bounds = []
        for _ in range(n_gaussians):
            bounds.append((0.0, 2.0))   # amplitude
            bounds.append((-0.25, 0.25))  # mean
            bounds.append((0.01, 0.25))   # std
        
        # Initial guess: symmetric Gaussians
        x0 = np.zeros(3 * n_gaussians)
        for i in range(n_gaussians):
            x0[3*i] = 1.0  # amp
            x0[3*i+1] = (i - (n_gaussians-1)/2) * 0.05  # spread out means
            x0[3*i+2] = 0.05  # std
        
        try:
            result = differential_evolution(
                _objective,
                bounds,
                args=(n_gaussians,),
                seed=42,
                maxiter=50,
                popsize=10,
                tol=1e-6,
                polish=False
            )
            
            if time.time() > deadline - 10:
                break
            
            h_de = _gaussian_mixture_to_heights(result.x, n_gaussians)
            val_de = compute_autoconv_max(h_de)
            
            if val_de < best_val:
                best_val = val_de
                best_h = h_de.copy()
            
            # L-BFGS-B polish
            if time.time() > deadline - 10:
                break
            
            result2 = minimize(
                _objective,
                result.x,
                args=(n_gaussians,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            
            if result2.success:
                h_lbfgs = _gaussian_mixture_to_heights(result2.x, n_gaussians)
                val_lbfgs = compute_autoconv_max(h_lbfgs)
                
                if val_lbfgs < best_val:
                    best_val = val_lbfgs
                    best_h = h_lbfgs.copy()
                    
        except Exception:
            continue
    
    # Final coordinate descent refinement on the best structure
    if time.time() < deadline - 5:
        current_h = best_h.copy()
        S1 = np.sum(current_h)
        S2 = np.sum(current_h**2)
        
        max_cd_iter = 50
        for iteration in range(max_cd_iter):
            if time.time() > deadline - 5:
                break
            
            improved = False
            hmax = np.max(current_h)
            
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                
                h_i_old = current_h[i]
                S1_other = S1 - h_i_old
                S2_other = S2 - h_i_old**2
                
                if S1_other < 1e-15:
                    continue
                
                # Case 1: h_i <= hmax
                h_i_case1 = -S1_other + np.sqrt(S1_other**2 + S2_other)
                h_i_case1 = max(0.0, h_i_case1)
                
                candidates = [h_i_old, h_i_case1, 0.0, h_i_old*0.5, h_i_old*1.5]
                candidates = [max(0.0, c) for c in candidates if c is not None and not np.isnan(c)]
                
                best_h_i = h_i_old
                best_f = compute_autoconv_max(current_h)
                
                for h_i_cand in candidates:
                    new_h = current_h.copy()
                    new_h[i] = h_i_cand
                    f_val = compute_autoconv_max(new_h)
                    if f_val < best_f:
                        best_f = f_val
                        best_h_i = h_i_cand
                
                if best_h_i != h_i_old:
                    current_h[i] = best_h_i
                    improved = True
                    S1 = np.sum(current_h)
                    S2 = np.sum(current_h**2)
            
            current_val = compute_autoconv_max(current_h)
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
            
            if not improved:
                break
    
    # Final check: ensure nonnegativity
    best_h = np.maximum(best_h, 0.0)
    
    # Return as list of floats
    return list(best_h)
