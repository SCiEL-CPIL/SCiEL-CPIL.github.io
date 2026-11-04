"""Second autocorrelation inequality -- SIDON-SET DIRECT OPTIMIZATION.

Strategy:
1. The objective R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) benefits from spreading
   mass over many indices to keep the autoconvolution peak low relative to the
   L2 norm, provided the indices form a "Sidon" set (unique pairwise sums) to
   prevent peak stacking.
2. We initialize h with a large Sidon set of indices (heights=1, others=0).
3. We perform direct L-BFGS-B optimization on the 50 non-negative heights to
   fine-tune the distribution, allowing for non-Sidon structures if they yield
   a higher ratio.
4. We normalize the final result to ||h||_1 = 1 to ensure numerical stability
   and consistency with scale-invariant shape optimization.
"""

import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
DH = 1.0 / 20.0
OFFSET = 10  # 1/2 / DH

_DEADLINE = 150.0
_SAFETY = 15.0


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _SAFETY - _elapsed()


def _ratio_from_h(h):
    """R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) from a precomputed nonneg h."""
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 1e-300 or l2sq <= 1e-300:
        return 0.0
    full = np.convolve(h, h, mode='full')
    ac = DH * full[OFFSET:OFFSET + N]
    acmax = float(np.max(ac))
    if acmax <= 1e-300:
        return 0.0
    return l2sq / (l1 * acmax)


def _logR(h):
    h = np.maximum(h, 0.0)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 1e-300 or l2sq <= 1e-300:
        return -1e30
    full = np.convolve(h, h, mode='full')
    ac = DH * full[OFFSET:OFFSET + N]
    acmax = float(np.max(ac))
    if acmax <= 1e-300:
        return -1e30
    R = l2sq / (l1 * acmax)
    if R <= 1e-300:
        return -1e30
    return math.log(R)


def _logR_and_grad(h):
    """Analytic gradient of log R w.r.t. h."""
    h_pos = np.maximum(h, 0.0)
    
    l2sq = float(np.dot(h_pos, h_pos))
    l1 = float(np.sum(h_pos))
    if l1 <= 1e-300 or l2sq <= 1e-300:
        return -1e30, np.zeros_like(h)

    full = np.convolve(h_pos, h_pos, mode='full')
    ac = DH * full[OFFSET:OFFSET + N]
    acmax = float(np.max(ac))
    if acmax <= 1e-300:
        return -1e30, np.zeros_like(h)

    R = l2sq / (l1 * acmax)
    logR = math.log(R)

    # d logR / d h_i
    # logR = log l2sq - log l1 - log acmax
    # d l2sq / dh_i = 2 h_i
    # d l1   / dh_i = 1
    # d acmax/ dh_i: only argmax index m matters.
    # ac_m = DH * sum_{j} h_j h_{m-j}
    # d ac_m / dh_i = 2 DH h_{m-i} (if valid)
    
    m = int(np.argmax(ac))
    dac_dh = np.zeros(N)
    j = m - np.arange(N)
    valid = (j >= 0) & (j < N)
    dac_dh[valid] = 2.0 * DH * h_pos[j[valid]]

    dlogR_dh = (2.0 * h_pos / l2sq) - (1.0 / l1) - (dac_dh / acmax)
    
    # Mask out where h was originally <= 0 (gradient is 0 there for L-BFGS-B bounds)
    mask = h > 0.0
    grad = np.where(mask, dlogR_dh, 0.0)
    
    if not np.all(np.isfinite(grad)):
        grad = np.zeros_like(h)
        
    return logR, grad


def _neg_logR(h):
    v, _ = _logR_and_grad(h)
    return -v


def _neg_logR_jac(h):
    v, g = _logR_and_grad(h)
    return v, -g


def _find_sidon_set(n, size):
    """Find a Sidon set of given size in {0, ..., n-1}."""
    # Greedy construction
    S = []
    for i in range(n):
        if len(S) >= size:
            break
        # Check if i can be added
        ok = True
        for j in S:
            s = i + j
            # Check if s is already a sum of two elements in S
            # We need to check if s = x+y for x,y in S
            # Efficient check: maintain set of sums
            pass
            
    # Simpler: Random search for a good set
    best_S = None
    best_score = 0
    rng = np.random.default_rng(42)
    
    for _ in range(1000):
        S = rng.choice(n, size=size, replace=False)
        S = sorted(S)
        # Calculate max overlap in autoconv
        h = np.zeros(n)
        h[S] = 1.0
        full = np.convolve(h, h, mode='full')
        ac = DH * full[OFFSET:OFFSET + N]
        acmax = np.max(ac)
        # We want to minimize acmax
        score = -acmax
        if score > best_score:
            best_score = score
            best_S = S
            
    return best_S


def construct():
    global _start
    _start = time.time()

    # 1. Generate initial Sidon-like sets
    # Try different sizes
    best_h = np.ones(N) / N  # Fallback: uniform
    best_log = _logR(best_h)
    
    sizes_to_try = [7, 8, 9, 10, 12, 15, 20]
    
    for size in sizes_to_try:
        if _time_left() < 10.0:
            break
            
        # Find a good Sidon set of this size
        S = _find_sidon_set(N, size)
        if S is None:
            continue
            
        h0 = np.zeros(N)
        h0[S] = 1.0
        
        # Optimize from this start
        try:
            res = minimize(
                _neg_logR, h0,
                method='L-BFGS-B',
                jac=_neg_logR_jac,
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-12,
                         'gtol': 1e-9, 'disp': False},
                callback=lambda x: (np.nan if (time.time() - _start) > _DEADLINE - _SAFETY else x),
            )
            
            if res.x is not None:
                h_opt = np.maximum(res.x, 0.0)
                # Normalize to L1=1 for consistent scale
                l1 = np.sum(h_opt)
                if l1 > 1e-10:
                    h_opt = h_opt / l1
                else:
                    continue
                    
                v = _logR(h_opt)
                if v > best_log:
                    best_log = v
                    best_h = h_opt.copy()
                    
        except Exception:
            pass

    # 2. Also try random sparse initializations
    rng = np.random.default_rng(20240501)
    for _ in range(20):
        if _time_left() < 10.0:
            break
        # Random sparse vector
        k = rng.integers(5, 25)
        indices = rng.choice(N, size=k, replace=False)
        h0 = np.zeros(N)
        h0[indices] = 1.0
        
        try:
            res = minimize(
                _neg_logR, h0,
                method='L-BFGS-B',
                jac=_neg_logR_jac,
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-12,
                         'gtol': 1e-9, 'disp': False},
                callback=lambda x: (np.nan if (time.time() - _start) > _DEADLINE - _SAFETY else x),
            )
            
            if res.x is not None:
                h_opt = np.maximum(res.x, 0.0)
                l1 = np.sum(h_opt)
                if l1 > 1e-10:
                    h_opt = h_opt / l1
                else:
                    continue
                    
                v = _logR(h_opt)
                if v > best_log:
                    best_log = v
                    best_h = h_opt.copy()
                    
        except Exception:
            pass

    # Final polish on the best found
    if _time_left() > 10.0:
        try:
            res = minimize(
                _neg_logR, best_h,
                method='L-BFGS-B',
                jac=_neg_logR_jac,
                bounds=[(0, None)] * N,
                options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-14,
                         'gtol': 1e-11, 'disp': False},
                callback=lambda x: (np.nan if (time.time() - _start) > _DEADLINE - _SAFETY else x),
            )
            if res.x is not None:
                h_opt = np.maximum(res.x, 0.0)
                l1 = np.sum(h_opt)
                if l1 > 1e-10:
                    h_opt = h_opt / l1
                v = _logR(h_opt)
                if v > best_log:
                    best_h = h_opt.copy()
        except Exception:
            pass

    # Ensure non-negative and finite
    best_h = np.maximum(best_h, 0.0)
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N) / N
        
    return [float(x) for x in best_h]


if __name__ == '__main__':
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _ratio_from_h(h))
    print("time =", time.time() - t0)
