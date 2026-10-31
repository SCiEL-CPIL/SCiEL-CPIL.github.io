import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # Fourier coefficient approach: minimize non-DC energy subject to h in [0,1]
    # h[i] = 1/2 + sum of Fourier perturbation terms
    # For even N: params are a_k, b_k for k=1..N/2-1, and a_{N/2}
    # Total N-1 parameters
    
    n_params = N - 1  # 99 for N=100
    
    def params_to_h(params):
        h = np.full(N, 0.5)
        for k in range(1, N // 2):
            idx = 2 * (k - 1)
            a = params[idx]
            b = params[idx + 1]
            h += a * np.cos(2 * np.pi * k * np.arange(N) / N)
            h += b * np.sin(2 * np.pi * k * np.arange(N) / N)
        # Nyquist term (k = N/2)
        a_half = params[2 * (N // 2 - 1)]
        h += a_half * np.cos(np.pi * np.arange(N))
        return h
    
    def objective(params):
        h = params_to_h(params)
        
        # Penalty for out-of-bounds
        penalty = np.sum(np.maximum(0.0, -h) ** 2 + np.maximum(0.0, h - 1.0) ** 2)
        
        h_clipped = np.clip(h, 0, 1)
        
        # Compute all cyclic overlaps
        overlaps = np.array([np.sum(h_clipped * (1 - np.roll(h_clipped, t))) for t in range(N)])
        max_overlap = np.max(overlaps)
        
        return max_overlap + 1000.0 * penalty
    
    # Multiple random restarts with decreasing scale
    best_h = np.full(N, 0.5)
    best_score = np.inf
    
    np.random.seed(123)
    for restart in range(5):
        scale = 0.1 / (restart + 1)
        x0 = np.random.randn(n_params) * scale
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                           bounds=[(-0.5, 0.5)] * n_params,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
            h = params_to_h(res.x)
            h = np.clip(h, 0, 1)
            
            # Compute score
            overlaps = np.array([np.sum(h * (1 - np.roll(h, t))) for t in range(N)])
            score = np.max(overlaps)
            
            if score < best_score:
                best_score = score
                best_h = h.copy()
        except Exception:
            pass
    
    h = best_h
    
    # Ensure sum is exactly N/2
    s = np.sum(h)
    if s > 0:
        h = h * (N / 2.0) / s
    h = np.clip(h, 0, 1)
    
    # Fine-tune sum to be exactly N/2
    diff = N / 2.0 - np.sum(h)
    idx = np.argsort(h)[::-1]
    for i in idx:
        if diff > 0:
            add = min(diff, 1.0 - h[i])
            h[i] += add
            diff -= add
        elif diff < 0:
            sub = min(-diff, h[i])
            h[i] -= sub
            diff += sub
        else:
            break
            
    return h
