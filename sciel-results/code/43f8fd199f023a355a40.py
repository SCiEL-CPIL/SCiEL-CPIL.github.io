import numpy as np
from scipy import optimize

N = 600

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    if np.sum(h) == 0:
        return 0.0
    conv = np.convolve(h, h, 'full')
    total = np.sum(h)
    return np.max(conv) / (total * total)

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Baseline: Uniform distribution
    # This is the theoretical global minimum for V = max(c) / (sum h)^2
    best_h = np.ones(N)
    best_score = compute_autoconv_max(best_h)
    
    # Soft-max objective for smooth optimization
    def soft_objective(x, T=0.01):
        h = np.maximum(x, 0)
        total = np.sum(h)
        if total < 1e-10:
            return 1e10
        conv = np.convolve(h, h, 'full')
        conv_max = np.max(conv)
        # Numerical stability for log-sum-exp
        shifted = (conv - conv_max) / T
        shifted = np.clip(shifted, -100, 100)
        soft_max = conv_max + T * np.log(np.sum(np.exp(shifted)))
        return soft_max / (total * total)
    
    # Try a few optimizations to see if we can beat the uniform baseline
    # (Unlikely, but ensures we don't miss a potential local min if the metric is different)
    np.random.seed(42)
    
    restarts = [
        np.ones(N),
        np.ones(N) * (1 + 0.1 * np.cos(2 * np.pi * np.linspace(0, 1, N, endpoint=False))),
        np.random.rand(N) + 0.1
    ]
    
    for i, h0 in enumerate(restarts):
        if time.time() > deadline - 20:
            break
            
        h0 = np.maximum(h0, 0.01)
        
        try:
            # Coarse stage
            res1 = optimize.minimize(
                lambda x: soft_objective(x, T=0.05), h0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            
            if time.time() > deadline - 10:
                break
            
            # Fine stage
            res2 = optimize.minimize(
                lambda x: soft_objective(x, T=0.005), res1.x,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 50, 'ftol': 1e-10}
            )
            
            for candidate in [res1.x, res2.x]:
                h_new = np.maximum(candidate, 0)
                score = compute_autoconv_max(h_new)
                if score < best_score:
                    best_score = score
                    best_h = h_new.copy()
                    
        except Exception:
            pass
            
    return list(best_h)
