import numpy as np
from scipy.optimize import minimize
import time
import warnings

N = 50
warnings.filterwarnings('ignore')

def _precompute_conv_matrix():
    """Precompute the autoconvolution kernel for step functions using vectorization."""
    dw = 1.0 / N
    x = np.linspace(-0.25, 0.25, N + 1)
    
    # Sample autoconvolution at fine grid points
    n_pts = 4 * N  # fine grid
    t_grid = np.linspace(-0.5, 0.5, n_pts + 1)
    midpoints = (t_grid[:-1] + t_grid[1:]) / 2.0
    
    # Vectorized computation of K
    xi = x[:-1].reshape(N, 1, 1)
    xi1 = x[1:].reshape(N, 1, 1)
    xj = x[:-1].reshape(1, N, 1)
    xj1 = x[1:].reshape(1, N, 1)
    t = midpoints.reshape(1, 1, -1)
    
    lo = np.maximum(xi, t - xj1)
    hi = np.minimum(xi1, t - xj)
    K = np.maximum(0, hi - lo)
    
    return K, midpoints

_K, _t_mid = _precompute_conv_matrix()
_K_flat = _K.reshape(N * N, -1)

def _compute_score(h):
    """Compute the ratio L2^2 / (L1 * sup|autoconv|) for step heights h."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    
    # Scale invariance: we can normalize h to sum to 1 to improve conditioning
    # Score is invariant under scaling, so let's work with normalized h
    h_norm = h / sum_h
    
    dw = 1.0 / N
    
    # L1 norm (integral) = 1 * dw (since sum h_norm = 1)
    L1 = 1.0 * dw
    L2_sq = np.sum(h_norm**2) * dw
    
    h_outer = np.outer(h_norm, h_norm).flatten()
    autoconv = h_outer @ _K_flat
    
    sup_conv = np.max(np.abs(autoconv))
    
    if sup_conv < 1e-15:
        return 0.0
    
    return L2_sq / (L1 * sup_conv)

def _score_neg(h):
    """Negative score for minimization."""
    return -_compute_score(h)

def _score_neg_smooth(h, beta=100.0):
    """Negative smooth surrogate score."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    
    h_norm = h / sum_h
    
    dw = 1.0 / N
    L1 = 1.0 * dw
    L2_sq = np.sum(h_norm**2) * dw
    
    h_outer = np.outer(h_norm, h_norm).flatten()
    autoconv = h_outer @ _K_flat
    
    abs_ac = np.abs(autoconv)
    max_val = np.max(abs_ac)
    if max_val < 1e-15:
        return 0.0
    
    shifted = beta * (abs_ac - max_val)
    log_sum = max_val + np.log(np.sum(np.exp(shifted))) / beta
    
    if log_sum < 1e-15:
        return 0.0
    
    return -(L2_sq / (L1 * log_sum))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = -np.inf
    
    # Generate diverse random starts
    # The seed program uses uniform or exponential starts.
    # The inspiration suggests that sparse structures (Sidon sets) might be better.
    # Let's try to generate starts that are sparse or have specific structures.
    
    # 1. Uniform start (baseline)
    h0 = np.ones(N)
    
    # 2. Sparse starts (random subsets of indices)
    # This is a heuristic based on the "Sidon set" insight.
    # We'll generate a few random sparse vectors.
    
    num_starts = 15
    
    for i in range(num_starts):
        if time.time() > deadline - 20:
            break
            
        if i == 0:
            h0 = np.ones(N)
        elif i == 1:
            # Sparse: 5 random peaks
            idx = np.random.choice(N, 5, replace=False)
            h0 = np.zeros(N)
            h0[idx] = 1.0
        elif i == 2:
            # Sparse: 3 random peaks
            idx = np.random.choice(N, 3, replace=False)
            h0 = np.zeros(N)
            h0[idx] = 1.0
        elif i == 3:
            # Sparse: 7 random peaks
            idx = np.random.choice(N, 7, replace=False)
            h0 = np.zeros(N)
            h0[idx] = 1.0
        elif i == 4:
            # Alternating
            h0 = np.zeros(N)
            h0[::2] = 1.0
        else:
            # Random exponential
            h0 = np.random.exponential(1.0, N)
        
        # Initial optimization with smooth surrogate
        try:
            result = minimize(_score_neg_smooth, h0, method='L-BFGS-B',
                             bounds=[(0, 10)] * N,
                             args=(50.0,),
                             options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6})
            h_trial = np.maximum(result.x, 0.0)
            
            # Polish with higher beta
            if time.time() < deadline - 10:
                result2 = minimize(_score_neg_smooth, h_trial, method='L-BFGS-B',
                                 bounds=[(0, 10)] * N,
                                 args=(200.0,),
                                 options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
                h_trial = np.maximum(result2.x, 0.0)
            
            # Final exact polish
            if time.time() < deadline - 5:
                result3 = minimize(_score_neg, h_trial, method='L-BFGS-B',
                                 bounds=[(0, 10)] * N,
                                 options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
                h_trial = np.maximum(result3.x, 0.0)
            
            score_trial = _compute_score(h_trial)
            if score_trial > best_score:
                best_score = score_trial
                best_h = h_trial.copy()
                
        except Exception:
            pass

    if best_h is None:
        # Fallback
        best_h = np.ones(N)
        
    return list(best_h)
