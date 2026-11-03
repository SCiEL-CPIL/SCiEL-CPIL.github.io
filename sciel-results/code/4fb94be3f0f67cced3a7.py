"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses Projected Gradient Descent with Barzilai-Borwein step sizes and 
adaptive smooth-max parameter. 
Projection onto the simplex-with-boxes is done via efficient water-filling.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 115  # Safety margin


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
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _smooth_max_val_grad(h, beta):
    """
    Smooth maximum of the linear overlap values and its gradient.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    # Avoid overflow
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights
    
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    for k in range(n):
        if w[k] == 0.0:
            continue
        factor = -(2.0 / n) * w[k]
        if k > 0:
            grad[k:] += factor * h[:n-k]
        if k < n:
            grad[:n-k] += factor * h[k:]
            
    return val, grad


def _project(h, target_sum):
    """
    Project h onto {x | sum(x) = target_sum, 0 <= x_i <= 1}.
    Uses bisection on the shift parameter tau.
    x_i = clip(h_i + tau, 0, 1)
    Find tau such that sum(x_i) = target_sum.
    """
    h = np.asarray(h, dtype=np.float64)
    
    def sum_proj(tau):
        return np.sum(np.clip(h + tau, 0.0, 1.0))
    
    lo = -max(np.max(h), 1.0) - 1.0
    hi = 1.0 - min(np.min(h), 0.0) + 1.0
    
    if sum_proj(lo) > target_sum:
        lo -= 10.0
    if sum_proj(hi) < target_sum:
        hi += 10.0
        
    for _ in range(50):
        mid = (lo + hi) / 2.0
        s = sum_proj(mid)
        if s < target_sum:
            lo = mid
        else:
            hi = mid
            
    tau = (lo + hi) / 2.0
    h_proj = np.clip(h + tau, 0.0, 1.0)
    
    diff = target_sum - np.sum(h_proj)
    if abs(diff) > 1e-12:
        mask = (h_proj > 1e-9) & (h_proj < 1.0 - 1e-9)
        if np.any(mask):
            h_proj[mask] += diff / np.sum(mask)
            h_proj = np.clip(h_proj, 0.0, 1.0)
            
    return h_proj


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    target_sum = N / 2.0

    def _update_best(h):
        nonlocal best_h, best_score
        h = _project(h, target_sum)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    def _optimize_slsqp(h0, beta=100.0):
        nonlocal best_h, best_score
        h = _project(h0.copy(), target_sum)
        
        def obj(x):
            if time.time() > _DEADLINE - 10:
                return 1e6
            val, _ = _smooth_max_val_grad(x, beta)
            return val

        def grad(x):
            if time.time() > _DEADLINE - 10:
                return np.zeros(N)
            _, g = _smooth_max_val_grad(x, beta)
            return g

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - target_sum, 'jac': lambda x: np.ones(N)})
        bounds = [(0, 1) for _ in range(N)]
        
        try:
            res = minimize(obj, h, method='SLSQP', jac=grad, bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-8})
            if res.success or res.fun < best_score:
                _update_best(res.x)
        except Exception:
            pass

    # 1. Cosine Start
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    _optimize_slsqp(h_cos)
    
    # 2. Random Starts
    if time.time() - t0 < _DEADLINE - 20:
        for _ in range(10):
            if time.time() - t0 > _DEADLINE - 10:
                break
            h_rand = np.random.uniform(0, 1, N)
            _optimize_slsqp(h_rand)
            
    # 3. Step Function Starts
    if time.time() - t0 < _DEADLINE - 10:
        h_step = np.zeros(N)
        h_step[:N//2] = 1.0
        _optimize_slsqp(h_step)
        
        h_step2 = np.zeros(N)
        h_step2[::2] = 1.0
        h_step2 = _project(h_step2, target_sum)
        _optimize_slsqp(h_step2)

    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _project(best_h, target_sum)
    return best_h
