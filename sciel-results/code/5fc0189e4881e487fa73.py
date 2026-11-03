"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Structural change: use scipy SLSQP with analytic gradients on the
log-sum-exp smooth surrogate, combined with a bisection on t (the
minimum achievable max overlap) for a feasibility-based approach.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_TARGET = N / 2.0
_NFFT = 256
_DEADLINE = time.time() + 140


def _autocorr(h):
    h_pad = np.zeros(_NFFT)
    h_pad[:N] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H))
    return corr[:N]


def _overlap_values(h):
    n = len(h)
    r = _autocorr(h)
    return (2.0 / n) * (n / 2.0 - r)


def _score(h):
    cs = _overlap_values(h)
    return float(np.max(cs))


def _logsumexp_grad(h, beta):
    """Returns (f, grad) for f = logsumexp(beta * C(h)) / beta."""
    cs = _overlap_values(h)
    m = np.max(cs)
    e = np.exp(beta * (cs - m))
    s = np.sum(e)
    val = (m + np.log(s)) / beta
    g_c = e / s  # softmax weights
    
    n = N
    g_h = np.zeros(n)
    factor = -(2.0 / n)
    for k in range(n):
        gc = g_c[k]
        if gc < 1e-15:
            continue
        if k == 0:
            g_h += factor * gc * (2.0 * h)
        else:
            g_h[k:] += factor * gc * h[:n - k]
            g_h[:n - k] += factor * gc * h[k:]
    return val, g_h


def _slsqp_optimize(h0, beta, time_budget=8.0):
    """Run SLSQP on the smooth surrogate with equality and bound constraints."""
    t_start = time.time()
    h0 = np.clip(h0, 0.0, 1.0)
    # Normalize sum
    diff = _TARGET - np.sum(h0)
    h0 = h0 + diff / N
    h0 = np.clip(h0, 0.0, 1.0)
    # Fix sum exactly
    diff = _TARGET - np.sum(h0)
    if abs(diff) > 1e-12:
        mask = (h0 > 1e-10) & (h0 < 1.0 - 1e-10)
        if np.sum(mask) > 0:
            h0[mask] += diff / np.sum(mask)
            h0 = np.clip(h0, 0.0, 1.0)
    
    def obj(x):
        if time.time() - t_start > time_budget:
            return 1e6
        v, _ = _logsumexp_grad(x, beta)
        return v
    
    def grad(x):
        if time.time() - t_start > time_budget:
            return np.zeros(N)
        _, g = _logsumexp_grad(x, beta)
        return g
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - _TARGET, 'jac': lambda x: np.ones(N)}
    ]
    bounds = [(0.0, 1.0)] * N
    
    try:
        res = minimize(obj, h0, jac=grad, method='SLSQP', bounds=bounds,
                       constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12,
                                                          'disp': False})
        if res.success or res.fun < 1e5:
            return res.x
    except Exception:
        pass
    return h0


def _project(h):
    """Exact Euclidean projection onto {h in [0,1]^N : sum(h) = N/2}."""
    h = np.array(h, dtype=np.float64, copy=True)
    target = _TARGET
    if abs(np.sum(h) - target) < 1e-13 and np.min(h) >= -1e-13 and np.max(h) <= 1 + 1e-13:
        return np.clip(h, 0.0, 1.0)
    
    lo = np.min(h) - 2.0
    hi = np.max(h) + 2.0
    
    def calc_sum(tau):
        return np.sum(np.clip(h - tau, 0.0, 1.0))
    
    if calc_sum(lo) < target:
        lo -= 10.0
    if calc_sum(hi) > target:
        hi += 10.0
    
    for _ in range(60):
        mid = (lo + hi) / 2.0
        s = calc_sum(mid)
        if abs(s - target) < 1e-14:
            lo = hi = mid
            break
        if s > target:
            lo = mid
        else:
            hi = mid
    
    tau = (lo + hi) / 2.0
    h_proj = np.clip(h - tau, 0.0, 1.0)
    diff = target - np.sum(h_proj)
    if abs(diff) > 1e-15:
        mask = (h_proj > 1e-10) & (h_proj < 1.0 - 1e-10)
        if np.sum(mask) > 0:
            h_proj[mask] += diff / np.sum(mask)
            h_proj = np.clip(h_proj, 0.0, 1.0)
        else:
            if diff > 0:
                idx = np.argmin(h_proj)
            else:
                idx = np.argmax(h_proj)
            h_proj[idx] += diff
            h_proj = np.clip(h_proj, 0.0, 1.0)
    return h_proj


def construct():
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
    # Initial candidates
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_const = np.full(N, 0.5)
    h_sin = 0.5 + 0.4 * np.sin(2.0 * np.pi * i / N)
    h_alt = np.where(i % 2 == 0, 0.8, 0.2)
    h_rand = np.random.uniform(0.1, 0.9, N)
    h_cos2 = 0.5 + 0.3 * np.cos(4.0 * np.pi * i / N)
    h_saw = 0.5 + 0.4 * (2.0 * (i / N - 0.5))
    h_rand2 = np.random.uniform(0.0, 1.0, N)
    h_rand3 = np.random.uniform(0.0, 1.0, N)
    
    candidates = [h_cos, h_const, h_sin, h_alt, h_rand, h_cos2, h_saw, h_rand2, h_rand3]
    for h0 in candidates:
        _update_best(h0)
    
    # Phase 1: SLSQP with moderate beta from multiple starts
    for h0 in candidates:
        if time.time() - t0 > _DEADLINE - 40:
            break
        h_opt = _slsqp_optimize(h0, beta=30.0, time_budget=8.0)
        _update_best(h_opt)
    
    # Phase 2: Higher beta refinement
    if best_h is not None and time.time() - t0 < _DEADLINE - 20:
        h_opt = _slsqp_optimize(best_h, beta=100.0, time_budget=10.0)
        _update_best(h_opt)
    
    # Phase 3: Very high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        h_opt = _slsqp_optimize(best_h, beta=300.0, time_budget=8.0)
        _update_best(h_opt)
    
    # Phase 4: Final polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 2:
        h_opt = _slsqp_optimize(best_h, beta=500.0, time_budget=4.0)
        _update_best(h_opt)
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _project(best_h)
    return best_h
