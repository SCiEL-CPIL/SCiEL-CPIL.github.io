import numpy as np
import time
import sys

N = 50

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|)"""
    h = np.asarray(h, dtype=float)
    if np.all(h < 1e-15):
        return 0.0
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _objective(params):
    """
    Objective function for scipy.optimize.
    params: [centers (K), sigmas (K), amps (K)]
    We want to maximize the ratio, so we minimize the negative.
    """
    K = len(params) // 3
    centers = params[:K]
    sigmas = params[K:2*K]
    amps = params[2*K:]
    
    # Ensure sigmas are positive
    sigmas = np.abs(sigmas) + 1e-6
    
    # Build h
    x = np.linspace(-0.25, 0.25, N)
    h = np.zeros(N)
    for c, s, a in zip(centers, sigmas, amps):
        h += a * np.exp(-0.5 * ((x - c) / s)**2)
        
    # h is non-negative by construction (sum of positive gaussians with positive amps)
    # Wait, amps can be negative? The problem says "nonnegative heights".
    # So we should enforce amps >= 0.
    # If we let scipy handle negative amps, h might have negative parts.
    # We should clip h to be non-negative? Or constrain amps?
    # Clipping h to non-negative is a valid operation if the "true" function was negative,
    # but it changes the geometry.
    # Better to constrain amps to be non-negative.
    # However, scipy minimize doesn't easily handle mixed bounds per parameter in a flat array
    # unless we use 'L-BFGS-B' or similar with bounds.
    
    # Let's assume for this objective function that we clip h to be non-negative.
    # This might lead to a suboptimal solution if the gaussian dips below zero,
    # but since we are summing positive gaussians (if amps > 0), h is always positive.
    # Let's enforce amps >= 0 in the optimization bounds.
    
    h = np.maximum(h, 0.0)
    
    return -_compute_ratio(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = np.zeros(N)
    best_ratio = 0.0
    
    # Try a few different numbers of Gaussians
    K_values = [1, 2, 3, 4, 5]
    
    # Bounds:
    # centers: [-0.25, 0.25]
    # sigmas: [0.001, 0.5] (max sigma is large enough to cover the domain)
    # amps: [0, 10] (arbitrary large upper bound)
    
    from scipy.optimize import minimize
    
    # We can use multiple initializations for each K
    np.random.seed(42)
    
    for K in K_values:
        if time.time() > deadline - 10.0:
            break
            
        # Try 5 random starts for this K
        for start_idx in range(5):
            if time.time() > deadline - 5.0:
                break
                
            # Random initialization
            centers = np.random.uniform(-0.2, 0.2, K)
            sigmas = np.random.uniform(0.01, 0.1, K)
            amps = np.random.uniform(0.1, 1.0, K)
            
            params0 = np.concatenate([centers, sigmas, amps])
            
            bounds = [(-0.25, 0.25)] * K + [(0.001, 0.5)] * K + [(0.0, 10.0)] * K
            
            try:
                res = minimize(_objective, params0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12})
                
                if res.success or res.fun < 0: # res.fun is -ratio, so < 0 means ratio > 0
                    params_opt = res.x
                    centers_opt = params_opt[:K]
                    sigmas_opt = np.abs(params_opt[K:2*K]) + 1e-6
                    amps_opt = params_opt[2*K:]
                    
                    x = np.linspace(-0.25, 0.25, N)
                    h_opt = np.zeros(N)
                    for c, s, a in zip(centers_opt, sigmas_opt, amps_opt):
                        h_opt += a * np.exp(-0.5 * ((x - c) / s)**2)
                    
                    h_opt = np.maximum(h_opt, 0.0)
                    
                    ratio = _compute_ratio(h_opt)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_h = h_opt.copy()
                        
            except Exception:
                continue
                
    # If no good result, fallback to uniform
    if best_ratio < 0.1:
        best_h = np.ones(N) / N
        best_ratio = _compute_ratio(best_h)
        
    # Final polish: Coordinate descent on the continuous h vector
    # This helps if the Gaussian approximation was slightly off or if local minima were found.
    # We can do a few steps of coordinate descent on the resulting h.
    
    # Simple coordinate descent: for each i, find optimal h[i] given others.
    # Since N=50, this is fast.
    
    def _optimize_single_coordinate(h, i):
        A = np.sum(h**2) - h[i]**2
        B = np.sum(h) - h[i]
        
        if B < 1e-12:
            return h[i]
            
        h_fixed = h.copy()
        h_fixed[i] = 0.0
        
        C = np.convolve(h_fixed, h_fixed, mode='full')
        
        D = np.zeros(2*N - 1)
        start_idx = i
        end_idx = i + N
        D[start_idx:end_idx] = 2.0 * h_fixed
        
        E = np.zeros(2*N - 1)
        if 2*i < 2*N - 1:
            E[2*i] = 1.0
            
        def ratio_func(x):
            if x < 0:
                x = 0.0
            l2_val = A + x*x
            l1_val = B + x
            if l1_val < 1e-15:
                return 0.0
            conv_vals = C + x * D + x*x * E
            sup_conv = np.max(conv_vals)
            if sup_conv < 1e-15:
                return 0.0
            return l2_val / (l1_val * sup_conv)
            
        # Golden Section Search
        gr = (np.sqrt(5) - 1) / 2
        a = 0.0
        b = max(1.0, 2.0 * np.sqrt(A + 1.0) + B)
        
        fa = ratio_func(a)
        fb = ratio_func(b)
        
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = ratio_func(c)
        fd = ratio_func(d)
        
        for _ in range(50):
            if fc < fd:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = ratio_func(d)
            else:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = ratio_func(c)
                
            if abs(b - a) < 1e-9:
                break
                
        x_opt = (a + b) / 2.0
        return x_opt

    # Run coordinate descent for a few passes
    coords = np.arange(N)
    for _ in range(20):
        if time.time() > deadline - 2.0:
            break
        np.random.shuffle(coords)
        for i in coords:
            new_val = _optimize_single_coordinate(best_h, i)
            best_h[i] = new_val
            
    final_ratio = _compute_ratio(best_h)
    if final_ratio > best_ratio:
        best_ratio = final_ratio
        
    return list(best_h)

if __name__ == "__main__":
    # Quick test
    h = construct()
    print(_compute_ratio(h))
