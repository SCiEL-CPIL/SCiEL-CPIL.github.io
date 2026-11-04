import numpy as np
from scipy.optimize import minimize

N = 50

def _autoconv_peak(h):
    """
    Compute the supremum norm of the autoconvolution of a piecewise-constant
    function defined by heights h on N equal intervals over [-1/4, 1/4].
    
    Using the analytic property that for piecewise constant functions on a grid
    with width w, the value at the centers of the sum-intervals is given by
    the discrete convolution of the heights.
    Specifically, peak = max_m sum_{i+j=m} h_i h_j.
    """
    # np.convolve(h, h, mode='full') gives coefficients of (sum h_i z^i)^2
    # which are exactly S_m = sum_{i+j=m} h_i h_j
    conv = np.convolve(h, h, mode='full')
    return np.max(conv)

def construct():
    # Objective function to minimize: - (||h||_2^2 / (||h||_1 * ||h*h||_inf))
    def objective(h_flat):
        h = np.maximum(0, h_flat)
        l1 = np.sum(h)
        if l1 < 1e-12:
            return 1e6
        l2_sq = np.sum(h**2)
        peak = _autoconv_peak(h)
        if peak < 1e-12:
            return 1e6
        ratio = l2_sq / (l1 * peak)
        return -ratio

    # Initial guesses
    # 1. Single spike
    h_spike = np.zeros(N)
    h_spike[N // 2] = 1.0
    
    # 2. Uniform
    h_uniform = np.ones(N)
    
    # 3. Random
    np.random.seed(42)
    h_random = np.random.rand(N)
    
    best_h = None
    best_score = -1e6
    
    candidates = [h_spike, h_uniform, h_random]
    
    for h0 in candidates:
        # Normalize initial guess
        h0_norm = h0 / np.sum(h0)
        
        try:
            # Use L-BFGS-B for speed and robustness with bounds
            res = minimize(objective, h0_norm, method='L-BFGS-B', 
                           bounds=[(0, None)] * N,
                           options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            h_opt = np.maximum(0, res.x)
            
            # Evaluate score
            l1 = np.sum(h_opt)
            l2_sq = np.sum(h_opt**2)
            peak = _autoconv_peak(h_opt)
            if l1 > 0 and peak > 0:
                score = l2_sq / (l1 * peak)
                if score > best_score:
                    best_score = score
                    best_h = h_opt
        except Exception:
            pass

    if best_h is None:
        best_h = h_spike
        
    # Normalize to sum=1 for stability (scale invariance)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
        
    return best_h
