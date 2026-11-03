"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines SLSQP (from current) with Nesterov Accelerated Gradient (from inspiration 1).
Uses a robust projection onto the constraint set and adaptive beta for smooth max."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 110  # Safety margin before 120s kill

def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    # r_k = sum_{i=k}^{n-1} h[i] * h[i-k] for k=0..n-1
    # This is the autocorrelation. Use FFT for O(n log n) instead of O(n^2).
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs

def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))

def _smooth_max_val_grad(h, beta):
    """
    Gradient of the smooth maximum objective (log-sum-exp approximation).
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    # Avoid overflow
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    
    # C_k = 1 - (2/n) * r_k
    # dC_k/dh_j = -(2/n) * ( h_{j-k} (if j>=k) + h_{j+k} (if j+k<n) )
    
    # Vectorized gradient computation
    for k in range(n):
        if w[k] < 1e-15:
            continue
        factor = -(2.0 / n) * w[k]
        
        # Term 1: h_{j-k} for j >= k
        if k > 0:
            grad[k:] += factor * h[:n-k]
        # Term 2: h_{j+k} for j+k < n
        if k < n:
            grad[:n-k] += factor * h[k:]
            
    return val, grad

def _project(h):
    """
    Project h onto the set {x : 0 <= x_i <= 1, sum(x) = N/2}.
    Uses binary search on the offset lambda.
    """
    h = np.asarray(h, dtype=np.float64)
    target = N / 2.0
    
    # If sum is already close, just clip and fix tiny error
    s = np.sum(h)
    if abs(s - target) < 1e-10:
        h = np.clip(h, 0.0, 1.0)
        # Final tiny correction
        diff = target - np.sum(h)
        if abs(diff) > 1e-15:
            # Adjust the element with most room
            if diff > 0:
                mask = h < 1.0 - 1e-10
                if np.any(mask):
                    idx = np.where(mask)[0][np.argmin(h[mask])]
                    h[idx] += diff
                    h = np.clip(h, 0.0, 1.0)
            else:
                mask = h > 1e-10
                if np.any(mask):
                    idx = np.where(mask)[0][np.argmax(h[mask])]
                    h[idx] += diff
                    h = np.clip(h, 0.0, 1.0)
        return h

    # Binary search for lambda such that sum(clip(h - lambda, 0, 1)) = target
    # f(lambda) = sum(clip(h - lambda, 0, 1)) is monotonically decreasing
    lo = np.min(h) - 1.0
    hi = np.max(h) + 1.0
    
    for _ in range(50):
        mid = (lo + hi) / 2.0
        proj = np.clip(h - mid, 0.0, 1.0)
        s = np.sum(proj)
        if s > target:
            lo = mid
        else:
            hi = mid
        
    proj = np.clip(h - (lo + hi) / 2.0, 0.0, 1.0)
    
    # Ensure exact sum in floating point
    diff = target - np.sum(proj)
    if abs(diff) > 1e-14:
        # Distribute error to elements not at bounds
        mask = (proj > 1e-12) & (proj < 1.0 - 1e-12)
        if np.any(mask):
            proj[mask] += diff / np.sum(mask)
            proj = np.clip(proj, 0.0, 1.0)
        else:
            # Fallback: adjust one element
            if diff > 0:
                idx = np.argmin(proj)
            else:
                idx = np.argmax(proj)
            proj[idx] += diff
            proj = np.clip(proj, 0.0, 1.0)
            
    return proj

def _nesterov_optimize(h0, beta_start, beta_end, max_iter, lr_start):
    """
    Nesterov Accelerated Gradient on the smooth max objective.
    """
    h = _project(h0.copy())
    h_prev = h.copy()
    t_k = 1.0
    lr = lr_start
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
            
        # Interpolate beta
        beta = beta_start + (beta_end - beta_start) * (it / max_iter)
        
        val, grad = _smooth_max_val_grad(h, beta)
        
        # Gradient step
        h_new = h - lr * grad
        
        # Project
        h_new = _project(h_new)
        
        # Nesterov update
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
        h_next = _project(h_next)
        
        h_prev = h
        h = h_next
        t_k = t_new
        
        # Adaptive learning rate decay
        lr *= 0.995
        
    return h

def _slsqp_optimize(h0, beta, maxiter, ftol):
    """
    SLSQP optimization for final polishing.
    """
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
             'jac': lambda x: np.ones(N)}]
    bounds = [(0.0, 1.0)] * N
    
    def obj_func(x):
        val, _ = _smooth_max_val_grad(x, beta)
        return val
        
    def jac_func(x):
        _, grad = _smooth_max_val_grad(x, beta)
        return grad
        
    try:
        res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol})
        if res.success:
            return res.x
        else:
            # Check if it improved
            s_res = _score(res.x)
            s_init = _score(h0)
            if s_res < s_init:
                return res.x
    except Exception:
        pass
    return h0

def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _project(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    i = np.arange(N)
    
    # 1. Generate diverse starting points
    starts = []
    
    # Cosine
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    starts.append(h_cos)
    
    # Sine
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    starts.append(h_sin)
    
    # Two-frequency
    h_2f = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / N) + 0.2 * np.cos(4.0 * np.pi * i / N)
    starts.append(h_2f)
    
    # Random
    for _ in range(5):
        h_r = np.random.uniform(0, 1, N)
        starts.append(h_r)
        
    # 2. Run Nesterov Accelerated Gradient from each start
    for h0 in starts:
        if time.time() > _DEADLINE - 20:
            break
        h_nag = _nesterov_optimize(h0, beta_start=20.0, beta_end=100.0, max_iter=300, lr_start=0.05)
        _update_best(h_nag)
        
    # 3. Polish the best result with SLSQP
    if best_h is not None and time.time() < _DEADLINE - 10:
        # First pass with moderate beta
        h_polish1 = _slsqp_optimize(best_h, beta=50.0, maxiter=100, ftol=1e-8)
        _update_best(h_polish1)
        
        # Second pass with higher beta for sharpness
        if time.time() < _DEADLINE - 5:
            h_polish2 = _slsqp_optimize(best_h, beta=150.0, maxiter=200, ftol=1e-10)
            _update_best(h_polish2)
            
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _project(best_h)
    return best_h
