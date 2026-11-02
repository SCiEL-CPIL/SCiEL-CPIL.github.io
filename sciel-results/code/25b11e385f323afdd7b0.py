"""Optimized program for second autocorrelation inequality."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _autoconv_peak(h):
    """Compute supremum norm of autoconvolution of step function heights."""
    # Autoconvolution: (f*f)(t) = integral f(s)f(t-s)ds
    # For step heights on equal intervals, this is a convolution
    n = len(h)
    # Full convolution length is 2*n - 1
    conv = np.convolve(h, h, mode='full')
    return np.max(conv)

def _score(h):
    """Compute the ratio ||f||_2^2 / (||f||_1 * ||f*f||_inf)"""
    h = np.array(h, dtype=float)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    peak = _autoconv_peak(h)
    if l1 < 1e-15 or peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * peak)

def _normalize_l1(h):
    """Normalize to L1=1."""
    s = np.sum(h)
    if s < 1e-15:
        return h
    return h / s

def construct():
    deadline = time.time() + 100  # Safety margin before 120s limit
    
    best_h = np.ones(N)  # Start with seed
    best_score = _score(best_h)
    
    # Strategy: Use sparse spikes with L1=1 normalization
    # Try various patterns of sparse nonnegative heights
    
    candidates = []
    
    # Single spike
    for pos in range(N):
        h = np.zeros(N)
        h[pos] = 1.0
        candidates.append(h)
    
    # Two spikes
    for i in range(N):
        for j in range(i+1, N):
            h = np.zeros(N)
            h[i] = 0.5
            h[j] = 0.5
            candidates.append(h)
    
    # Three spikes
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                h = np.zeros(N)
                h[i] = 1.0/3
                h[j] = 1.0/3
                h[k] = 1.0/3
                candidates.append(h)
    
    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline:
            break
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Try to improve with local optimization on the best sparse solution
    if time.time() < deadline - 30:
        # Use the best sparse pattern as starting point
        # Optimize heights (allowing continuous values) with L1 constraint
        def neg_score(params):
            h = np.abs(params)  # Enforce nonnegativity
            h = _normalize_l1(h)
            return -_score(h)
        
        x0 = best_h.copy()
        # Add small noise to break symmetry
        x0 = x0 + np.random.uniform(-0.01, 0.01, N)
        x0 = np.abs(x0)
        x0 = _normalize_l1(x0)
        
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if time.time() < deadline:
                h_opt = np.abs(res.x)
                h_opt = _normalize_l1(h_opt)
                s_opt = _score(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final polish: try more aggressive local search
    if time.time() < deadline - 10:
        # Coordinate descent
        h = best_h.copy()
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                old_val = h[i]
                # Try adjusting h[i]
                for delta in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1]:
                    new_val = old_val + delta
                    if new_val < 0:
                        continue
                    h_new = h.copy()
                    h_new[i] = new_val
                    h_new = _normalize_l1(h_new)
                    s_new = _score(h_new)
                    if s_new > _score(h):
                        h = h_new
                        improved = True
                        break
            if not improved:
                break
        s_final = _score(h)
        if s_final > best_score:
            best_score = s_final
            best_h = h.copy()
    
    return list(best_h)
