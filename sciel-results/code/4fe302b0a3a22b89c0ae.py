"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.

Strategy:
1. Start from a strong analytic 2-block initial guess (golden-ratio split).
2. Refine with SLSQP using a smooth max (log-sum-exp) objective and analytic gradient.
3. Add a final local discrete polish (coordinate descent on individual entries).
4. Use a deadline guard to return the best feasible result before the 150s kill.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0  # safety margin before 150s kill

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _fix_sum(h):
    """Adjust h so that sum(h) = N/2, keeping values in [0,1]."""
    target = N / 2.0
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if s < 1e-10:
        return np.full(N, 0.5)
    # Scale to target sum, then clip and iteratively fix
    h = h * (target / s)
    h = np.clip(h, 0.0, 1.0)
    for _ in range(20):
        diff = target - h.sum()
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h < 1.0
            if mask.any():
                h[mask] += diff / mask.sum()
            else:
                break
        else:
            mask = h > 0.0
            if mask.any():
                h[mask] += diff / mask.sum()
            else:
                break
        h = np.clip(h, 0.0, 1.0)
    return h

def _smooth_max(x, alpha=200.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0 / alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _overlap_vector(h):
    """Compute the overlap vector h_i * (1 - h_{i+1}) for i=0..N-2, and h_{N-1} for last."""
    n = len(h)
    ov = np.empty(n)
    ov[:-1] = h[:-1] * (1.0 - h[1:])
    ov[-1] = h[-1]
    return ov

def _objective_and_grad(h, alpha=200.0):
    """Objective: smooth max of overlap. Returns (value, gradient)."""
    ov = _overlap_vector(h)
    smax = _smooth_max(ov, alpha)
    
    # Gradient of smooth max w.r.t. overlap
    x_max = np.max(ov)
    exps = np.exp(alpha * (ov - x_max))
    w = exps / np.sum(exps)
    
    # Gradient w.r.t. h
    grad = np.zeros(N)
    for i in range(N - 1):
        if w[i] > 1e-15:
            # ov[i] = h[i] * (1 - h[i+1])
            # d(ov[i])/d(h[i]) = 1 - h[i+1]
            # d(ov[i])/d(h[i+1]) = -h[i]
            grad[i] += w[i] * (1.0 - h[i + 1])
            grad[i + 1] += w[i] * (-h[i])
    if w[N - 1] > 1e-15:
        # ov[N-1] = h[N-1]
        grad[N - 1] += w[N - 1] * 1.0
    
    return smax, grad

def _two_block_init():
    """Analytic 2-block initial guess using golden ratio split."""
    A = (3.0 - math.sqrt(5.0)) / 2.0  # ~0.381966
    B = 1.0 - A  # ~0.618034
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    h0 = _fix_sum(h0)
    return h0

def _local_polish(h, max_iters=500):
    """Coordinate descent polish: adjust individual entries to reduce max overlap."""
    h = h.copy()
    best_score = _score(h)
    
    for _ in range(max_iters):
        improved = False
        for i in range(N):
            # Try small perturbations
            orig = h[i]
            for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                h[i] = orig + delta
                if h[i] < 0.0 or h[i] > 1.0:
                    h[i] = orig
                    continue
                h = _fix_sum(h)
                s = _score(h)
                if s < best_score - 1e-10:
                    best_score = s
                    improved = True
                    break
            h[i] = orig if not improved else h[i]
            if not improved:
                h[i] = orig
        if not improved:
            break
    
    return h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Phase 1: Start from analytic 2-block init
    h_best = _two_block_init()
    best_score = _score(h_best)
    
    # Also try linear ramp as another starting point
    h_ramp = np.linspace(1.0, 0.0, N)
    h_ramp = _fix_sum(h_ramp)
    s_ramp = _score(h_ramp)
    if s_ramp < best_score:
        h_best = h_ramp.copy()
        best_score = s_ramp
    
    # Phase 2: SLSQP optimization with smooth max objective
    try:
        if time.time() < deadline - 20.0:
            constraints = [{'type': 'eq', 'fun': lambda h: h.sum() - N / 2.0}]
            bounds = [(0.0, 1.0)] * N
            
            res = minimize(
                lambda h: _objective_and_grad(h)[0],
                h_best,
                jac=lambda h: _objective_and_grad(h)[1],
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            
            if res.x is not None:
                h_slsqp = _fix_sum(res.x)
                s_slsqp = _score(h_slsqp)
                if s_slsqp < best_score:
                    h_best = h_slsqp
                    best_score = s_slsqp
    except Exception:
        pass
    
    # Phase 3: Local discrete polish
    try:
        if time.time() < deadline - 10.0:
            h_polished, s_polished = _local_polish(h_best, max_iters=300)
            if s_polished < best_score:
                h_best = h_polished
                best_score = s_polished
    except Exception:
        pass
    
    # Final: ensure feasibility
    h_best = _fix_sum(h_best)
    
    return h_best
