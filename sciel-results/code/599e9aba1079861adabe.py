import numpy as np
from scipy.optimize import minimize
import time
import warnings

N = 50
_TIME_LIMIT = 135.0
warnings.filterwarnings('ignore')

def _precompute_conv_kernel():
    """
    Precompute the autoconvolution kernel for step functions on [-1/4, 1/4].
    The step function f(t) has height h_i on interval [x_i, x_{i+1}].
    The autoconvolution (f*f)(t) = integral f(tau) f(t-tau) d tau.
    This is a piecewise quadratic function? No, for step functions (rectangles),
    the convolution of two rectangles is a trapezoid/triangle (piecewise linear).
    Wait, the convolution of two step functions is the sum of convolutions of pairs of rectangles.
    Each pair (i, j) contributes h_i * h_j * (convolution of rect_i and rect_j).
    The convolution of two unit-width rectangles shifted to positions x_i and x_j
    is a trapezoid centered at x_i + x_j.
    Specifically, let R_i(t) = 1 if t in [x_i, x_{i+1}] else 0.
    (R_i * R_j)(t) is the length of intersection of [x_i, x_{i+1}] and [t-x_{j+1}, t-x_j].
    This function is piecewise linear.
    The total autoconvolution is a sum of such piecewise linear functions, hence piecewise linear.
    The maximum of a piecewise linear function occurs at a vertex (breakpoint).
    The breakpoints are at x_i + x_j and x_i + x_{j+1} (and symmetric).
    Since x_i are on a uniform grid, the breakpoints are on a uniform grid with step size 2/N?
    Actually, the grid points for t are sums of grid points.
    To be safe, we can sample on a fine grid and take the max.
    However, the "inspiration" used a precomputed matrix K where K[i,j,k] is the overlap length.
    Then autoconv[t_k] = sum_{i,j} h_i h_j K[i,j,k].
    This allows vectorization.
    
    Let's use a grid for t. The support of f is [-1/4, 1/4].
    Support of f*f is [-1/2, 1/2].
    Let's use a grid of size M = 4*N or 8*N for accuracy.
    """
    dw = 1.0 / N
    x = np.linspace(-0.25, 0.25, N + 1)
    
    # Grid for t
    M = 8 * N
    t_grid = np.linspace(-0.5, 0.5, M + 1)
    t_mid = (t_grid[:-1] + t_grid[1:]) / 2.0
    
    # We need to compute K[i, j, k] = length of intersection of [x_i, x_{i+1}] and [t_mid[k] - x_{j+1}, t_mid[k] - x_j]
    # This is max(0, min(x_{i+1}, t_k - x_j) - max(x_i, t_k - x_{j+1}))
    
    # Vectorized computation
    # i: N, j: N, k: M
    # Memory: N*N*M = 50*50*400 = 1,000,000 floats. 8MB. Acceptable.
    
    i_idx = np.arange(N)[:, None, None]
    j_idx = np.arange(N)[None, :, None]
    k_idx = np.arange(M)[None, None, :]
    
    x_i = x[i_idx]
    x_i1 = x[i_idx + 1]
    x_j = x[j_idx]
    x_j1 = x[j_idx + 1]
    t_k = t_mid[k_idx]
    
    # Intersection length
    lo = np.maximum(x_i, t_k - x_j1)
    hi = np.minimum(x_i1, t_k - x_j)
    
    K = np.maximum(0.0, hi - lo)
    
    # Flatten K for matrix multiplication
    # We want autoconv[k] = sum_{i,j} h_i h_j K[i,j,k]
    # Let H = h outer h (N*N, 1)
    # Let K_flat = K.reshape(N*N, M)
    # autoconv = K_flat.T @ H ?
    # autoconv[k] = sum_{i,j} K[i,j,k] * h[i] * h[j]
    # This is equivalent to (K_flat.T @ (h*h)) ? No.
    # Let v = h * h (element-wise, length N*N)
    # autoconv = K_flat.T @ v
    # Wait, K_flat is (N*N, M). v is (N*N,).
    # K_flat.T @ v gives (M,).
    # This works.
    
    return K.reshape(N * N, M), t_mid, dw

_K_flat, _t_mid, _dw = _precompute_conv_kernel()

def _compute_score(h):
    """
    Compute the ratio L2^2 / (L1 * sup|autoconv|).
    h: array of heights.
    """
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    
    s_h = np.sum(h)
    if s_h < 1e-15:
        return 0.0
    
    # L1 norm of step function is sum(h_i) * dw
    L1 = s_h * _dw
    
    # L2 norm squared is sum(h_i^2) * dw
    L2_sq = np.dot(h, h) * _dw
    
    # Autoconvolution
    # v = h * h (outer product flattened)
    v = np.outer(h, h).flatten()
    autoconv = _K_flat.T @ v
    
    # Supremum norm
    sup_conv = np.max(np.abs(autoconv))
    
    if sup_conv < 1e-15:
        return 0.0
    
    return L2_sq / (L1 * sup_conv)

def _score_neg(h):
    return -_compute_score(h)

def _score_neg_smooth(h, beta=100.0):
    """
    Smooth surrogate using log-sum-exp for the supremum.
    """
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    
    s_h = np.sum(h)
    if s_h < 1e-15:
        return 0.0
    
    L1 = s_h * _dw
    L2_sq = np.dot(h, h) * _dw
    
    v = np.outer(h, h).flatten()
    autoconv = _K_flat.T @ v
    
    abs_ac = np.abs(autoconv)
    max_val = np.max(abs_ac)
    if max_val < 1e-15:
        return 0.0
    
    # Soft max
    # log(sum(exp(beta * (x - max)))) / beta + max
    shifted = beta * (abs_ac - max_val)
    # Clip to avoid overflow, though shifted <= 0
    log_sum = max_val + np.log(np.sum(np.exp(shifted))) / beta
    
    if log_sum < 1e-15:
        return 0.0
    
    return -(L2_sq / (L1 * log_sum))

def _refine(h, max_iter=100, lr=0.01):
    """
    Simple gradient ascent refinement using numerical gradients.
    """
    h = np.array(h, dtype=float)
    h = np.maximum(h, 0.0)
    best_h = h.copy()
    best_score = _compute_score(h)
    
    for _ in range(max_iter):
        if np.sum(h) < 1e-15:
            break
        
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros(N)
        for i in range(N):
            h_plus = h.copy()
            h_plus[i] += eps
            h_minus = h.copy()
            h_minus[i] = max(0, h[i] - eps)
            
            s_plus = _compute_score(h_plus)
            s_minus = _compute_score(h_minus)
            
            if h[i] > eps:
                grad[i] = (s_plus - s_minus) / (2 * eps)
            else:
                grad[i] = (s_plus - s_minus) / eps
        
        h = h + lr * grad
        h = np.maximum(h, 0.0)
        
        # Normalize to prevent drift
        if np.sum(h) > 1e6:
            h = h / np.sum(h) * N
        
        cur_score = _compute_score(h)
        if cur_score > best_score:
            best_score = cur_score
            best_h = h.copy()
        else:
            # If no improvement, shrink step size
            lr *= 0.9
            if lr < 1e-6:
                break
                
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT - 5.0  # Safety margin
    
    # Initial guesses
    # The seed is [2/3, 1/3]
    h0 = np.zeros(N)
    h0[0] = 2.0 / 3.0
    h0[1] = 1.0 / 3.0
    
    best_h = h0.copy()
    best_score = _compute_score(best_h)
    
    inits = [
        h0.copy(),
        np.array([1.0/2, 1.0/2] + [0.0]*(N-2)),
        np.array([1.0/3]*3 + [0.0]*(N-3)),
        np.array([0.5, 0.3, 0.2] + [0.0]*(N-3)),
        np.ones(N) / N,
        np.array([1.0] + [0.0]*(N-1)),
    ]
    
    # Phase 1: Smooth optimization
    for init in inits:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(_score_neg_smooth, init, method='L-BFGS-B',
                           bounds=[(0, 10)] * N,
                           args=(50.0,),
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            h_trial = np.maximum(res.x, 0.0)
            s_trial = _compute_score(h_trial)
            if s_trial > best_score:
                best_score = s_trial
                best_h = h_trial
        except Exception:
            pass
    
    # Phase 2: Sharper smooth optimization
    for beta in [100.0, 200.0, 500.0]:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(_score_neg_smooth, best_h, method='L-BFGS-B',
                           bounds=[(0, 10)] * N,
                           args=(beta,),
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            h_trial = np.maximum(res.x, 0.0)
            s_trial = _compute_score(h_trial)
            if s_trial > best_score:
                best_score = s_trial
                best_h = h_trial
        except Exception:
            pass
    
    # Phase 3: Exact optimization (non-smooth)
    if time.time() < deadline - 10:
        try:
            res = minimize(_score_neg, best_h, method='L-BFGS-B',
                           bounds=[(0, 10)] * N,
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            h_trial = np.maximum(res.x, 0.0)
            s_trial = _compute_score(h_trial)
            if s_trial > best_score:
                best_score = s_trial
                best_h = h_trial
        except Exception:
            pass
    
    # Phase 4: Gradient refinement
    if time.time() < deadline - 5:
        try:
            h_ref, s_ref = _refine(best_h, max_iter=50, lr=0.05)
            if s_ref > best_score:
                best_score = s_ref
                best_h = h_ref
        except Exception:
            pass
            
    return [float(x) for x in best_h]
