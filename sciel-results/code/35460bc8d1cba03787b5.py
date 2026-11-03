"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Approach: Binary parity-constrained sequence h[2i] = 1 - h[2i+1].
Optimize the binary variables x[i] = h[2i] using Simulated Annealing to minimize
the maximum linear overlap C_k. Refine with continuous optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 110  # Safety margin before 120s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _fix_sum(h):
    """Adjust h to have sum exactly N/2, keeping values in [0, 1]."""
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h


def _construct_binary_sa():
    """
    Construct a binary sequence of length N with parity constraint h[2i] + h[2i+1] = 1.
    Uses Simulated Annealing to minimize the maximum overlap.
    """
    half = N // 2  # 50
    np.random.seed(42)
    
    # Initialize x
    x = np.random.randint(0, 2, half).astype(np.float64)
    
    def _build_h(x):
        h = np.zeros(N)
        h[0::2] = x
        h[1::2] = 1.0 - x
        return h
    
    def _cost(x):
        h = _build_h(x)
        cs = _linear_overlap_values(h)
        return float(np.max(cs))
    
    current_x = x.copy()
    current_cost = _cost(current_x)
    best_x = current_x.copy()
    best_cost = current_cost
    
    # Simulated Annealing parameters
    T_start = 1.0
    T_end = 1e-4
    alpha = 0.99
    T = T_start
    max_iter = 5000
    
    t0 = time.time()
    
    for i in range(max_iter):
        if time.time() - t0 > _DEADLINE - 60:
            break
        
        # Pick a random index to flip
        idx = np.random.randint(half)
        old_val = current_x[idx]
        current_x[idx] = 1.0 - old_val
        
        new_cost = _cost(current_x)
        delta = new_cost - current_cost
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_x = current_x.copy()
        else:
            current_x[idx] = old_val
            
        T = max(T * alpha, T_end)
        
    h = _build_h(best_x)
    return h, best_cost


def _refine_continuous(h0):
    """
    Refine a binary parity-constrained solution by allowing small continuous deviations
    while maintaining the parity structure approximately.
    """
    from scipy.optimize import minimize
    
    half = N // 2
    
    def _param_to_h(params):
        """params: 2*half values. h[2i] = params[i], h[2i+1] = params[half+i]"""
        h = np.zeros(N)
        h[0::2] = params[:half]
        h[1::2] = params[half:]
        return h
    
    def _h_to_params(h):
        params = np.zeros(2 * half)
        params[:half] = h[0::2]
        params[half:] = h[1::2]
        return params
    
    h0 = np.asarray(h0, dtype=np.float64)
    # Enforce parity constraint approximately: average with its parity-flipped version
    h_parity = np.zeros(N)
    h_parity[0::2] = (h0[0::2] + (1.0 - h0[1::2])) / 2.0
    h_parity[1::2] = (h0[1::2] + (1.0 - h0[0::2])) / 2.0
    h0 = np.clip(h_parity, 0.0, 1.0)
    
    x0 = _h_to_params(h0)
    
    def obj(x):
        h = _param_to_h(x)
        h = np.clip(h, 0.0, 1.0)
        cs = _linear_overlap_values(h)
        return float(np.max(cs))
    
    bounds = [(0.0, 1.0)] * (2 * half)
    
    try:
        # Use L-BFGS-B which handles bounds well
        res = minimize(obj, x0, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.fun < obj(x0):
            h_res = _param_to_h(res.x)
            h_res = np.clip(h_res, 0.0, 1.0)
            return h_res
    except Exception:
        pass
    
    return h0


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    
    best_h = None
    best_score = float('inf')
    
    # Try a few different random seeds for the SA construction
    for seed in [42, 1, 7, 13, 100]:
        if time.time() - t0 > _DEADLINE - 40:
            break
        
        np.random.seed(seed)
        h_try, score_try = _construct_binary_sa()
        h_try = _fix_sum(h_try)
        s = _score(h_try)
        
        if s < best_score:
            best_score = s
            best_h = h_try.copy()
            
    # If SA failed or didn't produce anything, fallback to random binary
    if best_h is None:
        np.random.seed(0)
        x = np.random.randint(0, 2, N // 2).astype(np.float64)
        best_h = np.zeros(N)
        best_h[0::2] = x
        best_h[1::2] = 1.0 - x
        best_h = _fix_sum(best_h)
        best_score = _score(best_h)
        
    # Refine the best solution continuously
    if time.time() - t0 < _DEADLINE - 10:
        h_refined = _refine_continuous(best_h)
        h_refined = _fix_sum(h_refined)
        s_ref = _score(h_refined)
        if s_ref < best_score:
            best_score = s_ref
            best_h = h_refined
    
    # Final fallback
    if best_h is None or len(best_h) != N:
        best_h = np.full(N, 0.5)
        
    best_h = _fix_sum(best_h)
    return best_h
