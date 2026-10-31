import numpy as np
import time

N = 100

def _max_overlap_linear(h):
    """Max over linear shifts k=0..N-1 of sum_{i} h[i] * (1-h[i+k]) where i+k<N."""
    one_minus_h = 1.0 - h
    max_ov = 0.0
    for k in range(N):
        m = N - k
        ov = np.sum(h[:m] * one_minus_h[k:])
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _max_overlap_linear_fast(h):
    """Faster using FFT for linear cross-correlation."""
    one_minus_h = 1.0 - h
    # Linear cross-correlation: (h * one_minus_h)[k] = sum_i h[i] * one_minus_h[i+k]
    # Use FFT: pad to size >= 2N-1
    size = 2 * N
    H = np.fft.rfft(h, size)
    O = np.fft.rfft(one_minus_h, size)
    corr = np.fft.irfft(H * np.conj(O), size)
    # corr[k] for k=0..N-1 gives sum_i h[i]*one_minus_h[i+k]
    return np.max(corr[:N])

def _max_overlap_smooth_linear(h, alpha=100.0):
    """Smooth max via log-sum-exp for linear shifts."""
    one_minus_h = 1.0 - h
    size = 2 * N
    H = np.fft.rfft(h, size)
    O = np.fft.rfft(one_minus_h, size)
    corr = np.fft.irfft(H * np.conj(O), size)
    corr = corr[:N]
    scaled = alpha * corr
    max_scaled = np.max(scaled)
    return (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))) / alpha

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap_linear_fast(best_h)
    
    from scipy.optimize import minimize
    
    def objective(x):
        h = np.clip(x, 0.0, 1.0)
        return _max_overlap_smooth_linear(h, alpha=100.0)
    
    def constraint_sum(x):
        return np.sum(x) - N / 2.0
    
    constraints = [
        {'type': 'eq', 'fun': constraint_sum},
    ]
    
    bounds = [(0.0, 1.0)] * N
    
    rng = np.random.default_rng(42)
    
    for restart in range(30):
        if time.time() > deadline - 20:
            break
        
        # Different initializations
        if restart == 0:
            h0 = np.full(N, 0.5)
        elif restart % 3 == 1:
            # Block pattern
            h0 = np.zeros(N)
            h0[::2] = 1.0
            h0 = h0 * (N/2) / np.sum(h0)
            h0 = np.clip(h0, 0, 1)
        elif restart % 3 == 2:
            # Random with noise
            h0 = rng.uniform(0, 1, N)
        else:
            # Sinusoidal
            t = np.linspace(0, 4*np.pi, N)
            h0 = 0.5 + 0.4*np.sin(t)
            h0 = np.clip(h0, 0, 1)
        
        # Normalize to sum N/2
        s = np.sum(h0)
        if s > 0:
            h0 = h0 * (N/2) / s
        h0 = np.clip(h0, 0.0, 1.0)
        # Fine adjust
        diff = N/2 - np.sum(h0)
        if abs(diff) > 1e-10:
            h0 = h0 + diff / N
            h0 = np.clip(h0, 0.0, 1.0)
        
        res = minimize(objective, h0, method='SLSQP', constraints=constraints,
                       bounds=bounds, options={'maxiter': 300, 'ftol': 1e-10})
        
        if time.time() > deadline - 10:
            break
        
        if res.success:
            h_sol = np.clip(res.x, 0.0, 1.0)
            h_sol = h_sol - (np.sum(h_sol) - N/2) / N
            h_sol = np.clip(h_sol, 0.0, 1.0)
            score = _max_overlap_linear_fast(h_sol)
            if score < best_score:
                best_score = score
                best_h = h_sol.copy()
    
    # Polish
    if time.time() < deadline - 5:
        def objective_tight(x):
            h = np.clip(x, 0.0, 1.0)
            return _max_overlap_smooth_linear(h, alpha=500.0)
        
        res = minimize(objective_tight, best_h, method='SLSQP', constraints=constraints,
                       bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12})
        if res.success:
            h_sol = np.clip(res.x, 0.0, 1.0)
            h_sol = h_sol - (np.sum(h_sol) - N/2) / N
            h_sol = np.clip(h_sol, 0.0, 1.0)
            score = _max_overlap_linear_fast(h_sol)
            if score < best_score:
                best_score = score
                best_h = h_sol.copy()
    
    best_h = best_h - (np.sum(best_h) - N/2) / N
    best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
