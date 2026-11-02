import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.array(h, dtype=float)
    if np.any(h < -1e-15):
        return -1.0  # invalid
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return -1.0
    l2_sq = np.sum(h**2)
    # Autoconvolution: full convolution of h with itself
    autocorr = np.convolve(h, h, mode='full')
    l_inf = np.max(autocorr)
    if l_inf < 1e-15:
        return -1.0
    return l2_sq / (l1 * l_inf)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin
    
    best_h = [1.0] * N  # seed
    best_ratio = compute_ratio(best_h)
    
    # Try some analytical candidates
    candidates = []
    
    # Spike
    spike = [0.0] * N
    spike[0] = 1.0
    candidates.append(spike)
    
    # Symmetric spike
    spike_sym = [0.0] * N
    spike_sym[N//2] = 1.0
    candidates.append(spike_sym)
    
    # Triangular
    tri = [1.0 - abs(i - (N-1)/2.0) / ((N-1)/2.0) for i in range(N)]
    candidates.append(tri)
    
    # Gaussian
    sigma = 5.0
    gauss = [np.exp(-((i - (N-1)/2.0)**2) / (2*sigma**2)) for i in range(N)]
    candidates.append(gauss)
    
    # Block of various sizes
    for k in [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50]:
        block = [0.0] * N
        for i in range(k):
            block[i] = 1.0
        candidates.append(block)
    
    # Symmetric block
    for k in [2, 4, 6, 8, 10, 14, 20, 30, 40]:
        sb = [0.0] * N
        center = N // 2
        half = k // 2
        for i in range(max(0, center-half), min(N, center+half+1)):
            sb[i] = 1.0
        candidates.append(sb)
    
    # Evaluate all candidates
    for c in candidates:
        r = compute_ratio(c)
        if r > best_ratio:
            best_ratio = r
            best_h = list(c)
    
    # Now do numerical optimization: maximize ratio with constraint sum(h) = 1
    # Parameterize h as exp(x) / sum(exp(x)) to ensure nonnegativity and sum=1
    # Actually, since scale doesn't matter, just maximize ratio with sum(h)=1
    
    def objective(x):
        # x are log-spacings or just raw values with constraint
        h = np.exp(x)  # ensures positive
        h = h / np.sum(h)  # normalize to sum=1
        return -compute_ratio(h)
    
    # Use differential evolution for global search
    try:
        if time.time() < deadline - 60.0:
            # Initial bounds
            x0 = np.zeros(N)
            bounds = [(-10, 10)] * N
            
            def de_obj(x):
                h = np.exp(x)
                h_sum = np.sum(h)
                if h_sum < 1e-15:
                    return 1e10
                h = h / h_sum
                r = compute_ratio(h)
                if r < 0:
                    return 1e10
                return -r
            
            result = differential_evolution(
                de_obj, bounds, 
                maxiter=100, popsize=15, 
                seed=42, tol=1e-8,
                polish=True
            )
            if time.time() < deadline - 5.0:
                h_opt = np.exp(result.x)
                h_opt = h_opt / np.sum(h_opt)
                r_opt = compute_ratio(h_opt)
                if r_opt > best_ratio:
                    best_ratio = r_opt
                    best_h = list(h_opt)
    except Exception:
        pass
    
    # Local refinement with Nelder-Mead if time permits
    try:
        if time.time() < deadline - 10.0:
            h_current = np.array(best_h)
            h_current = np.maximum(h_current, 0)
            if np.sum(h_current) > 1e-15:
                h_current = h_current / np.sum(h_current)
            
            def lm_obj(x):
                h = np.exp(x)
                h_sum = np.sum(h)
                if h_sum < 1e-15:
                    return 1e10
                h = h / h_sum
                r = compute_ratio(h)
                if r < 0:
                    return 1e10
                return -r
            
            x_start = np.log(np.maximum(h_current, 1e-15))
            result2 = minimize(lm_obj, x_start, method='Nelder-Mead', 
                              options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            if time.time() < deadline - 2.0:
                h_refined = np.exp(result2.x)
                h_refined = h_refined / np.sum(h_refined)
                r_refined = compute_ratio(h_refined)
                if r_refined > best_ratio:
                    best_ratio = r_refined
                    best_h = list(h_refined)
    except Exception:
        pass
    
    # Final check: ensure nonnegative
    best_h = [max(0.0, h) for h in best_h]
    
    # If all zero, fall back to seed
    if sum(best_h) < 1e-15:
        best_h = [1.0] * N
    
    return best_h
