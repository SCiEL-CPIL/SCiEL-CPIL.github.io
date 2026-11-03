"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses FFT-accelerated overlap computation with a multi-strategy optimization:
1. Nesterov accelerated projected gradient on log-sum-exp smooth max
2. SLSQP refinement with epigraph formulation
3. Coordinate descent polish
4. Multi-restart with diverse initializations
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_TARGET = N / 2.0
# Next power of 2 >= 2*N for FFT
_NFFT = 1
while _NFFT < 2 * N:
    _NFFT *= 2

# Evaluator kills at 150s. We target finishing by 142s.
_DEADLINE = time.time() + 142


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    h_pad = np.zeros(_NFFT)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=_NFFT)
    corr = np.fft.irfft(H * np.conj(H), n=_NFFT)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _project_to_box_sum(h):
    """
    Project h onto {x : 0 <= x_i <= 1, sum(x) = N/2}.
    Uses iterative water-filling.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = _TARGET
    for _ in range(50):
        diff = target - np.sum(h)
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h < 1.0 - 1e-12
        else:
            mask = h > 1e-12
        if not np.any(mask):
            break
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    # Final exact correction
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h


def _smooth_max_obj_grad(h, beta):
    """
    Smooth max of C_k via log-sum-exp and its gradient.
    Returns (value, gradient).
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    
    if sum_exp < 1e-300:
        w = np.full(n, 1.0 / n)
        val = m
    else:
        w = exp_terms / sum_exp
        val = m + np.log(sum_exp) / beta
    
    # Gradient: dC_k/dh_j = -(2/n) * (h_{j-k} if j>=k + h_{j+k} if j+k<n)
    grad = np.zeros(n)
    for k in range(n):
        if w[k] < 1e-14:
            continue
        factor = -(2.0 / n) * w[k]
        if k > 0:
            grad[k:] += factor * h[:n - k]
        if k < n:
            grad[:n - k] += factor * h[k:]
    
    return val, grad


def _nesterov_pgd(h0, beta, n_iter, lr=None, time_budget=10.0):
    """
    Nesterov accelerated projected gradient descent on smooth max.
    """
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _project_to_box_sum(h)
    y = h.copy()
    
    t0 = time.time()
    best_h = h.copy()
    best_val = _score(h)
    
    for it in range(n_iter):
        if time.time() - t0 > time_budget:
            break
        
        # Nesterov momentum
        t_k = 1.0 + it
        t_k1 = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        alpha = (t_k - 1.0) / t_k1
        
        y = h + alpha * (h - y)
        
        val, grad = _smooth_max_obj_grad(y, beta)
        
        if lr is None:
            lr = 1.0 / (beta * 4.0)
        
        h_new = y - lr * grad
        h = _project_to_box_sum(h_new)
        
        cur_score = _score(h)
        if cur_score < best_val:
            best_val = cur_score
            best_h = h.copy()
        
        y = h
    
    return best_h


def _slsqp_epigraph(h0, beta, maxiter=300, ftol=1e-10, time_budget=20.0):
    """
    SLSQP optimization using epigraph formulation.
    Minimize t subject to smooth_max(C) - t <= 0, sum(h) = N/2, 0<=h<=1.
    """
    t0 = time.time()
    n = len(h0)
    init_t = _score(h0) + 0.01
    
    def obj(x):
        return x[n]
    
    def obj_grad(x):
        g = np.zeros(n + 1)
        g[n] = 1.0
        return g
    
    # Inequality constraint: smooth_max(C) - t <= 0
    def ineq_fun(x):
        h = x[:n]
        val, _ = _smooth_max_obj_grad(h, beta)
        return val - x[n]
    
    def ineq_grad(x):
        h = x[:n]
        _, g = _smooth_max_obj_grad(h, beta)
        gr = np.zeros(n + 1)
        gr[:n] = g
        gr[n] = -1.0
        return gr
    
    cons = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[:n]) - _TARGET,
         'jac': lambda x: np.concatenate([np.ones(n), [0.0]])},
        {'type': 'ineq', 'fun': ineq_fun, 'jac': ineq_grad}
    ]
    bounds = [(0.0, 1.0)] * n + [(0.0, 2.0)]
    
    try:
        res = minimize(obj, np.concatenate([h0.copy(), [init_t]]),
                       jac=obj_grad, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol})
        if res.x is not None and time.time() - t0 < time_budget:
            return res.x[:n]
    except Exception:
        pass
    return h0


def _coordinate_descent(h, n_sweeps=20, time_budget=10.0):
    """
    Coordinate descent polish: for each coordinate, find optimal value
    given others fixed, using the max overlap objective.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    t0 = time.time()
    best_score = _score(h)
    
    for sweep in range(n_sweeps):
        if time.time() - t0 > time_budget:
            break
        improved = False
        for j in range(N):
            if time.time() - t0 > time_budget:
                break
            # Try a small grid around current value
            orig = h[j]
            # Determine range to search
            lo = max(0.0, orig - 0.05)
            hi = min(1.0, orig + 0.05)
            # Also try endpoints
            candidates = np.unique(np.concatenate([
                [lo, hi, orig],
                np.linspace(lo, hi, 11)
            ]))
            best_c = orig
            best_s = best_score
            
            for c in candidates:
                h[j] = c
                # Adjust sum
                diff = _TARGET - np.sum(h)
                if abs(diff) > 1e-14:
                    # Distribute to other elements
                    others = np.arange(N) != j
                    if np.sum(others) > 0:
                        h[others] += diff / np.sum(others)
                        h = np.clip(h, 0.0, 1.0)
                        # Re-fix
                        h = _project_to_box_sum(h)
                    else:
                        h[j] = orig
                        continue
                s = _score(h)
                if s < best_s - 1e-12:
                    best_s = s
                    best_c = c
            
            h[j] = best_c
            h = _project_to_box_sum(h)
            if abs(h[j] - orig) > 1e-10:
                improved = True
            if best_s < best_score:
                best_score = best_s
        
        if not improved:
            break
    
    return h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _project_to_box_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Generate diverse initial candidates
    i = np.arange(N)
    candidates = []
    
    # Constant
    candidates.append(np.full(N, 0.5))
    
    # Cosine wave
    candidates.append(0.5 + 0.5 * np.cos(2.0 * np.pi * i / N))
    
    # Sine
    candidates.append(0.5 + 0.4 * np.sin(2.0 * np.pi * i / N))
    
    # Alternating
    candidates.append(np.where(i % 2 == 0, 0.8, 0.2))
    
    # Random
    candidates.append(np.random.uniform(0.1, 0.9, N))
    
    # Chirp-like
    candidates.append(0.5 + 0.3 * np.sin(2.0 * np.pi * i * (1 + i / N) / N))
    
    # Two-frequency
    candidates.append(0.5 + 0.3 * np.cos(2.0 * np.pi * i / N) + 0.2 * np.cos(4.0 * np.pi * i / N))
    
    # Random with structure
    for _ in range(3):
        h_r = np.random.uniform(0.2, 0.8, N)
        # Smooth it
        kernel = np.array([1, 2, 3, 2, 1], dtype=float)
        kernel /= kernel.sum()
        h_r = np.convolve(h_r, kernel, mode='same')
        candidates.append(h_r)
    
    # Evaluate all candidates
    for h0 in candidates:
        _update_best(h0)
    
    # Phase 1: Nesterov PGD with moderate beta from multiple starts
    if best_h is not None and time.time() - t0 < _DEADLINE - 60:
        for h0 in candidates[:5]:
            if time.time() - t0 > _DEADLINE - 50:
                break
            h_opt = _nesterov_pgd(h0, beta=20.0, n_iter=500, lr=None, time_budget=8.0)
            _update_best(h_opt)
    
    # Phase 2: Nesterov PGD with higher beta from best
    if best_h is not None and time.time() - t0 < _DEADLINE - 40:
        h_opt = _nesterov_pgd(best_h, beta=50.0, n_iter=800, lr=None, time_budget=10.0)
        _update_best(h_opt)
    
    # Phase 3: SLSQP epigraph refinement
    if best_h is not None and time.time() - t0 < _DEADLINE - 30:
        h_opt = _slsqp_epigraph(best_h, beta=50.0, maxiter=400, ftol=1e-10, time_budget=20.0)
        _update_best(h_opt)
    
    # Phase 4: Nesterov with high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 20:
        h_opt = _nesterov_pgd(best_h, beta=100.0, n_iter=1000, lr=None, time_budget=10.0)
        _update_best(h_opt)
    
    # Phase 5: SLSQP with high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 15:
        h_opt = _slsqp_epigraph(best_h, beta=100.0, maxiter=500, ftol=1e-12, time_budget=12.0)
        _update_best(h_opt)
    
    # Phase 6: Coordinate descent polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_opt = _coordinate_descent(best_h, n_sweeps=30, time_budget=8.0)
        _update_best(h_opt)
    
    # Phase 7: Final Nesterov with very high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 2:
        h_opt = _nesterov_pgd(best_h, beta=300.0, n_iter=500, lr=None, time_budget=5.0)
        _update_best(h_opt)
    
    # Phase 8: Emergency coordinate polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 0.5:
        h_opt = _coordinate_descent(best_h, n_sweeps=10, time_budget=3.0)
        _update_best(h_opt)
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _project_to_box_sum(best_h)
    return best_h
