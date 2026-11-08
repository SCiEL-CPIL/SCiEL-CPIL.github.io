import numpy as np
from scipy.optimize import minimize
import time

N = 100
dx = 2.0 / N

def _compute_overlaps(h):
    """Compute all circular overlaps O_k = sum h_j (1-h_{j+k}) dx."""
    n = len(h)
    H = np.fft.rfft(h, n=2*n)
    autocorr = np.fft.irfft(H * np.conj(H), n=2*n)[:n]
    # autocorr[k] = sum_j h[j] * h[(j-k)%n]
    # We want C[k] = sum_j h[j] * h[(j+k)%n]
    # C[k] = autocorr[(-k)%n]
    C = np.zeros(n)
    C[0] = autocorr[0]
    if n > 1:
        C[1:] = autocorr[n-1:0:-1]
    # O_k = (sum h_j - C_k) * dx = (N/2 - C_k) * dx
    overlaps = (n/2.0 - C) * dx
    return overlaps

def _objective(h, beta):
    """Soft-max objective with penalty for sum constraint."""
    overlaps = _compute_overlaps(h)
    max_val = np.max(overlaps)
    overlaps_shifted = overlaps - max_val
    # Use log-sum-exp for stability
    log_sum_exp = max_val + np.log(np.sum(np.exp(beta * overlaps_shifted))) / beta
    
    # Penalty for sum constraint
    sum_h = np.sum(h)
    target_sum = N / 2.0
    penalty = 1000.0 * (sum_h - target_sum) ** 2
    
    return log_sum_exp + penalty

def _project(h):
    """Project h to [0,1] and sum=N/2."""
    h = np.clip(h, 0.0, 1.0)
    current_sum = np.sum(h)
    target_sum = N / 2.0
    diff = target_sum - current_sum
    if abs(diff) > 1e-12:
        if diff > 0:
            mask = h < 1.0 - 1e-12
            if np.sum(mask) > 0:
                h[mask] += diff / np.sum(mask)
        else:
            mask = h > 1e-12
            if np.sum(mask) > 0:
                h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_h = np.full(N, 0.5)
    best_score = np.max(_compute_overlaps(best_h))
    
    np.random.seed(42)
    
    # List of betas for soft-max
    betas = [50, 100, 500, 1000, 5000]
    
    # Initializations: Constant, Random Binary-like, Random Uniform
    inits = [
        np.full(N, 0.5),
        np.random.choice([0.0, 1.0], N).astype(float),
        np.random.uniform(0, 1, N)
    ]
    # Ensure sum constraint for inits
    inits = [_project(h) for h in inits]
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
        
        for init in inits:
            if time.time() > deadline - 10:
                break
            
            # Add noise for restarts
            for restart in range(3):
                if time.time() > deadline - 10:
                    break
                
                h_init = init.copy()
                if restart > 0:
                    noise = np.random.normal(0, 0.2, N)
                    h_init += noise
                    h_init = _project(h_init)
                
                def obj(h):
                    return _objective(h, beta)
                
                bounds = [(0.0, 1.0) for _ in range(N)]
                
                try:
                    res = minimize(obj, h_init, method='L-BFGS-B', bounds=bounds, 
                                   options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
                    
                    h_opt = _project(res.x)
                    score = np.max(_compute_overlaps(h_opt))
                    
                    if score < best_score:
                        best_score = score
                        best_h = h_opt.copy()
                except Exception:
                    pass

    # Final polish with very high beta
    if time.time() < deadline - 5:
        beta_final = 10000
        def obj_final(h):
            return _objective(h, beta_final)
        
        bounds = [(0.0, 1.0) for _ in range(N)]
        
        try:
            res = minimize(obj_final, best_h, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            h_opt = _project(res.x)
            score = np.max(_compute_overlaps(h_opt))
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
        except Exception:
            pass

    # Ensure validity
    best_h = np.clip(best_h, 0.0, 1.0)
    if abs(np.sum(best_h) - N/2) > 1e-6:
        best_h = _project(best_h)
        
    return best_h
