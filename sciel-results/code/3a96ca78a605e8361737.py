"""Optimized program for second autocorrelation inequality."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
W = 1.0 / N  # step width: interval [-1/4, 1/4] has length 1/2, so each step = 1/100
# Actually interval is [-1/4, 1/4], length = 1/2, N=50 steps, width = 1/100
W = 0.5 / N

def _autoconv_sup(h):
    """Compute supremum of autoconvolution of step function h on [-1/4,1/4]."""
    # h is array of N step heights
    # Autoconvolution of step function: piecewise linear
    # At t=0: sum_i h_i^2 * W
    # The autoconvolution is maximized at t=0 for nonnegative h
    # But let's compute properly: (h*h)(t) for step function
    # For step function with values h_i on [x_{i-1}, x_i), 
    # (h*h)(t) = sum over pairs where intervals overlap shifted by t
    # Maximum is at t=0: sum_i h_i^2 * W
    return np.sum(h**2) * W

def _score(h):
    """Compute the ratio: L2^2 / (L1 * sup_autoconv)"""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0)  # ensure nonnegative
    l1 = np.sum(h) * W
    l2_sq = np.sum(h**2) * W
    if l1 < 1e-15:
        return 0.0
    sup_ac = _autoconv_sup(h)
    if sup_ac < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_ac)

def _neg_score(h):
    return -_score(h)

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Scale-invariant: fix sum to 1 for optimization
    # Maximize L2^2 / sup_autoconv subject to sum(h)*W = 1, h >= 0
    # Equivalent: maximize sum(h^2) / sup_autoconv with sum(h) = 1/W
    
    target_sum = 1.0 / W  # so L1 = 1
    
    def _objective(x):
        h = np.maximum(x, 0)
        # Project to sum constraint approximately
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        h = h * (target_sum / s)
        return -_score(h)
    
    # Multiple random restarts
    rng = np.random.default_rng(42)
    n_restarts = 200
    restarts_done = 0
    
    for r in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Initialize: various strategies
        if r == 0:
            h0 = np.ones(N) * (target_sum / N)
        elif r == 1:
            # Triangular
            h0 = np.linspace(0, 2, N) * (target_sum / N)
            h0 = np.maximum(h0, 0)
        elif r == 2:
            # Spike in middle
            h0 = np.zeros(N)
            h0[N//2] = target_sum
        elif r == 3:
            # Two spikes
            h0 = np.zeros(N)
            h0[N//4] = target_sum / 2
            h0[3*N//4] = target_sum / 2
        elif r == 4:
            # Uniform with noise
            h0 = np.ones(N) * (target_sum / N) * (1 + 0.5 * rng.standard_normal(N))
            h0 = np.maximum(h0, 0)
        else:
            # Random Dirichlet-like
            h0 = rng.dirichlet(np.ones(N) * 0.5) * target_sum
        
        h0 = np.maximum(h0, 0)
        s = np.sum(h0)
        if s > 0:
            h0 = h0 * (target_sum / s)
        
        # Optimize
        try:
            res = minimize(_objective, h0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.success or res.fun < 0:
                h_opt = np.maximum(res.x, 0)
                s_opt = np.sum(h_opt)
                if s_opt > 0:
                    h_opt = h_opt * (target_sum / s_opt)
                sc = _score(h_opt)
                if sc > best_score:
                    best_score = sc
                    best_h = h_opt.copy()
        except Exception:
            pass
        
        restarts_done += 1
    
    # Polish: try to improve with tighter tolerance
    if time.time() < deadline - 2:
        try:
            res = minimize(_objective, best_h, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < -best_score + 1e-10:
                h_polish = np.maximum(res.x, 0)
                s_p = np.sum(h_polish)
                if s_p > 0:
                    h_polish = h_polish * (target_sum / s_p)
                if _score(h_polish) > best_score:
                    best_h = h_polish
        except Exception:
            pass
    
    return list(best_h)
