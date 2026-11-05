import numpy as np
from scipy.optimize import minimize
import time

N = 400

def compute_value(h):
    h = np.asarray(h, dtype=np.float64)
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max_conv(h, beta):
    """Soft-max of autoconvolution using log-sum-exp for numerical stability."""
    conv = np.convolve(h, h)
    m = np.max(conv)
    if m > 1e10:
        return m
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    return m + np.log(np.sum(np.exp(shifted))) / beta

def _smooth_objective(params, beta, deadline, best_val_ref):
    """Smooth objective: minimize soft-max of autoconv / sum(h)^2 * 2*N"""
    if time.time() > deadline:
        return best_val_ref[0]
    h = params.copy()
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    soft_max = _soft_max_conv(h, beta)
    val = abs(2.0 * N * soft_max) / (s * s)
    if not np.isfinite(val):
        return 1e10
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Try several analytical patterns first
    patterns = []
    patterns.append(('ones', np.ones(N)))
    patterns.append(('ramp', np.linspace(1, 2, N)))
    
    x = np.linspace(0, 1, N)
    patterns.append(('quad', 1 + x))
    patterns.append(('quad2', 1 + x*x))
    
    for k in [1, 2, 3, 4, 5, 10, 20, 50]:
        for a in [0.1, 0.3, 0.5, 0.7, 0.9]:
            h = 1.0 + a * np.cos(2.0 * np.pi * k * np.arange(N) / N)
            patterns.append((f'cos_{k}_{a}', h))
        for a in [-0.1, -0.3, -0.5, -0.7, -0.9]:
            h = 1.0 + a * np.cos(2.0 * np.pi * k * np.arange(N) / N)
            patterns.append((f'cos_{k}_{a}', h))
    
    for k in [1, 2, 3, 4, 5, 10, 20]:
        for a in [0.3, 0.5, 0.7]:
            h = 1.0 + a * np.sin(2.0 * np.pi * k * np.arange(N) / N)
            patterns.append((f'sin_{k}_{a}', h))
    
    for block_frac in [0.1, 0.2, 0.3, 0.5, 0.7]:
        block_len = int(N * block_frac)
        h = np.zeros(N)
        h[:block_len] = 1.0
        patterns.append((f'block_{block_frac}', h))
    
    for gap_frac in [0.1, 0.2, 0.3]:
        gap_len = int(N * gap_frac)
        h = np.ones(N)
        h[N//2 - gap_len//2:N//2 + gap_len//2] = -0.5
        patterns.append((f'gap_{gap_frac}', h))
    
    # Two-level patterns
    for level2 in [-2.0, -1.0, -0.5, 0.5, 2.0]:
        for frac in [0.1, 0.2, 0.3, 0.5]:
            h = np.ones(N)
            n_ones = int(N * frac)
            h[:n_ones] = level2
            patterns.append((f'two_level_{level2}_{frac}', h))
    
    # Evaluate all patterns
    for name, h in patterns:
        val = compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Gradient-based optimization using soft-max surrogate
    if time.time() < deadline - 30:
        h0 = best_h.copy()
        if np.sum(h0) < 0:
            h0 = -h0
        
        best_val_ref = [best_val]
        
        def obj(params):
            return _smooth_objective(params, 50.0, deadline, best_val_ref)
        
        try:
            result = minimize(obj, h0, method='L-BFGS-B',
                            options={'maxiter': 200, 'maxfun': 500, 
                                    'ftol': 1e-12, 'gtol': 1e-8,
                                    'maxls': 50})
            
            if result.fun < best_val:
                best_val = result.fun
                best_h = result.x.copy()
        except Exception:
            pass
        
        # Try with different beta values (annealing)
        for beta in [20.0, 10.0, 5.0]:
            if time.time() > deadline - 20:
                break
            
            def obj_beta(params, b=beta):
                return _smooth_objective(params, b, deadline, best_val_ref)
            
            try:
                result = minimize(obj_beta, best_h, method='L-BFGS-B',
                                options={'maxiter': 100, 'maxfun': 300, 
                                        'ftol': 1e-12, 'gtol': 1e-8,
                                        'maxls': 30})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_h = result.x.copy()
            except Exception:
                pass
    
    # Final polish: coordinate descent on individual elements
    if time.time() < deadline - 5:
        improved = True
        max_rounds = 5
        round_num = 0
        while improved and round_num < max_rounds and time.time() < deadline - 5:
            improved = False
            round_num += 1
            for i in range(0, N, max(1, N // 100)):
                if time.time() > deadline - 5:
                    break
                for delta in [-0.1, -0.01, 0.01, 0.1]:
                    h_trial = best_h.copy()
                    h_trial[i] += delta
                    val = compute_value(h_trial)
                    if val < best_val - 1e-12:
                        best_val = val
                        best_h = h_trial
                        improved = True
    
    # Final check
    final_val = compute_value(best_h)
    if final_val < best_val:
        best_val = final_val
    
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
    
    return best_h
