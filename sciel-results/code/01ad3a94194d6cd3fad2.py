import numpy as np
from scipy.optimize import minimize

N = 100

def _score(h):
    h = np.asarray(h, dtype=np.float64)
    if h.shape[0] != N:
        return 1e9
    # Overlap at shift k: sum_i h[i] * (1 - h[(i+k) % N])
    # = N/2 - sum_i h[i]*h[(i+k)%N]
    # We want to minimize the max over k, i.e., minimize N/2 - min_k autocorr(k)
    # So maximize min_k autocorr(k).
    # autocorr(k) = sum_i h[i] * h[(i+k) % N]
    # Use FFT for speed: autocorr is circular autocorrelation
    # autocorr(k) = ifft(fft(h) * conj(fft(h)))
    H = np.fft.fft(h)
    ac = np.real(np.fft.ifft(H * np.conj(H)))
    # ac[k] = sum_i h[i] * h[(i-k) % N] ... actually circular autocorr
    # The max overlap = N/2 - min_k ac[k]
    return N / 2.0 - np.min(ac)

def _neg_score(h):
    return _score(h)

def construct():
    # Start from constant solution
    h0 = np.full(N, 0.5)
    
    # Use SLSQP with bounds [0,1] and equality constraint sum(h) = N/2
    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}]
    
    # Try multiple random starts
    best_h = h0.copy()
    best_val = _score(h0)
    
    rng = np.random.default_rng(42)
    
    for trial in range(20):
        if trial == 0:
            x0 = h0.copy()
        else:
            # Random perturbation
            x0 = rng.uniform(0.0, 1.0, N)
            # Normalize to sum = N/2
            s = np.sum(x0)
            if s > 0:
                x0 = x0 * (N / 2.0) / s
            # Clip to [0,1] and re-normalize iteratively
            for _ in range(10):
                x0 = np.clip(x0, 0.0, 1.0)
                s = np.sum(x0)
                if abs(s - N / 2.0) < 1e-12:
                    break
                x0 = x0 + (N / 2.0 - s) / N
            x0 = np.clip(x0, 0.0, 1.0)
        
        try:
            res = minimize(_neg_score, x0, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 500, 'ftol': 1e-10})
            if res.success:
                h_opt = np.clip(res.x, 0.0, 1.0)
                # Enforce exact sum
                h_opt = h_opt + (N / 2.0 - np.sum(h_opt)) / N
                h_opt = np.clip(h_opt, 0.0, 1.0)
                val = _score(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    return best_h
