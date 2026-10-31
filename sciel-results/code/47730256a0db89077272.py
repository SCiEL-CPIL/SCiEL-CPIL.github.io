"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines chaotic structure search with gradient-based refinement.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100
TARGET_SUM = N / 2.0

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _fix_sum(h, target=None):
    """Adjust h so that sum(h) == target (default N/2), preserving [0,1] bounds."""
    if target is None:
        target = TARGET_SUM
    h = np.asarray(h, dtype=float)
    diff = target - h.sum()
    if abs(diff) > 1e-12:
        h = h + diff / len(h)
        h = np.clip(h, 0.0, 1.0)
        # If clipping broke the sum, do a second pass on non-saturated elements
        diff2 = target - h.sum()
        if abs(diff2) > 1e-12:
            mask = (h > 1e-9) & (h < 1.0 - 1e-9)
            if np.any(mask):
                h[mask] += diff2 / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
                # Third pass if necessary
                diff3 = target - h.sum()
                if abs(diff3) > 1e-9:
                    mask2 = (h > 0) & (h < 1.0)
                    if np.any(mask2):
                        h[mask2] += diff3 / np.sum(mask2)
                        h = np.clip(h, 0.0, 1.0)
    return h

def _chaotic_h(params):
    """
    Generate h from logistic map parameters: [r, x0, offset, scale]
    r: control parameter (3.5 to 4.0 for chaos)
    x0: initial condition
    offset: vertical shift
    scale: scaling factor
    """
    r, x0, offset, scale = params
    x = np.zeros(N)
    x[0] = x0
    for i in range(1, N):
        x[i] = r * x[i-1] * (1.0 - x[i-1])
    h = scale * x + offset
    return _fix_sum(h)

def _objective_chaotic(params):
    h = _chaotic_h(params)
    return _score(h)

def _cosine_h(params):
    """
    Generate h from cosine parameters: [amp, offset, freq, phase]
    """
    amp, offset, freq, phase = params
    x = np.arange(N)
    h = amp * np.cos(2 * np.pi * freq * x / N + phase) + offset
    return _fix_sum(h)

def _objective_cosine(params):
    h = _cosine_h(params)
    return _score(h)

def _smooth_max_and_grad(x, beta):
    """
    Compute soft max of x and its gradient using log-sum-exp.
    Returns (value, gradient_vector).
    """
    x = np.asarray(x, dtype=float)
    max_x = np.max(x)
    exps = np.exp(beta * (x - max_x))
    sum_exps = np.sum(exps)
    weights = exps / sum_exps
    val = max_x + np.log(sum_exps) / beta
    return val, weights

def _overlap_grad(h, beta):
    """
    Compute gradient of smooth max of overlap w.r.t. h.
    overlap_i = h_i * (1 - h_{i+1}) for i < N-1
    overlap_{N-1} = h_{N-1}
    """
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    
    val, weights = _smooth_max_and_grad(overlap, beta)
    
    grad = np.zeros(n)
    for i in range(n):
        w = weights[i]
        if w == 0.0:
            continue
        if i < n - 1:
            # overlap_i = h_i * (1 - h_{i+1})
            # d(overlap_i)/d(h_i) = (1 - h_{i+1})
            # d(overlap_i)/d(h_{i+1}) = -h_i
            grad[i] += w * (1.0 - h[i+1])
            grad[i+1] += w * (-h[i])
        else:
            # overlap_{N-1} = h_{N-1}
            # d(overlap_{N-1})/d(h_{N-1}) = 1
            grad[i] += w * 1.0
    
    return val, grad

def _run_slsqp(h0, beta, maxiter, ftol, deadline):
    """
    Run SLSQP optimization with smooth max objective and its gradient.
    """
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - TARGET_SUM,
             'jac': lambda x: np.ones(N)}]
    bounds = [(0.0, 1.0)] * N
    
    def obj_func(x):
        val, _ = _overlap_grad(x, beta)
        return val
        
    def jac_func(x):
        _, grad = _overlap_grad(x, beta)
        return grad
        
    try:
        res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol})
        if res.success or res.fun < float('inf'):
            h_new = _fix_sum(res.x)
            return h_new, _score(h_new)
    except Exception:
        pass
    return None, float('inf')

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10s margin before 150s kill
    np.random.seed(42)
    
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    # 1. Chaotic structure search
    if time.time() < deadline - 40:
        rng = np.random.default_rng(42)
        for _ in range(15):
            p0 = np.array([
                rng.uniform(3.5, 4.0),  # r: chaotic region
                rng.uniform(0.1, 0.9),  # x0: initial condition
                rng.uniform(-0.5, 0.5), # offset
                rng.uniform(0.5, 1.5)   # scale
            ])
            res = minimize(_objective_chaotic, p0, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_h = _chaotic_h(res.x)
    
    # 2. Cosine structure search (from current program)
    if time.time() < deadline - 40:
        rng = np.random.default_rng(123)
        for _ in range(15):
            p0 = np.array([
                rng.uniform(0.2, 0.5),
                rng.uniform(0.3, 0.7),
                rng.uniform(0.5, 3.0),
                rng.uniform(0, 2*np.pi)
            ])
            res = minimize(_objective_cosine, p0, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_h = _cosine_h(res.x)
    
    # 3. Gradient-based refinement with SLSQP
    if time.time() < deadline - 20:
        # Phase 1: Coarse refinement
        h_opt, s_opt = _run_slsqp(best_h, beta=50.0, maxiter=200, ftol=1e-8, deadline=deadline)
        if h_opt is not None and s_opt < best_score:
            best_score = s_opt
            best_h = h_opt
    
    # 4. Final polish with higher beta
    if time.time() < deadline - 5:
        h_opt, s_opt = _run_slsqp(best_h, beta=100.0, maxiter=300, ftol=1e-10, deadline=deadline)
        if h_opt is not None and s_opt < best_score:
            best_score = s_opt
            best_h = h_opt
    
    # 5. Local search refinement
    if time.time() < deadline - 5:
        rng = np.random.default_rng(999)
        h = best_h.copy()
        for _ in range(3000):
            if time.time() > deadline - 2:
                break
            idx = rng.integers(0, N)
            step = rng.uniform(-0.05, 0.05)
            h_trial = h.copy()
            h_trial[idx] = np.clip(h_trial[idx] + step, 0.0, 1.0)
            h_trial = _fix_sum(h_trial)
            sc = _score(h_trial)
            if sc < best_score - 1e-10:
                best_score = sc
                h = h_trial
        best_h = h
    
    return _fix_sum(best_h)

if __name__ == "__main__":
    h = construct()
    print(_score(h))
