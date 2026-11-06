"""
Improved program for the third autocorrelation inequality task.

Strategy:
- Use smooth log-sum-exp surrogate for the max of autoconvolution.
- Full-dimensional gradient descent with analytic gradients and adaptive beta.
- Multiple restarts from diverse initializations (random, DCT-based, chirp-based).
- Final polish with very tight beta and many iterations.
- Deadline guard ensures we return before 150s wall-clock.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
_t0 = time.time()
_DEADLINE = 145.0  # safety margin under 150s wall-clock

# Precompute common arrays
_i_arr = np.arange(N, dtype=float)

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _fft_convolve(h):
    """Fast autoconvolution via FFT."""
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * n - 1]
    return conv

def _smooth_max_value(h, beta=80.0):
    """Smooth approximation of max of autoconvolution using log-sum-exp."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    log_sum_exp = m + np.log(np.sum(np.exp(scaled))) / beta
    return abs(2.0 * N * log_sum_exp / (s * s))

def _smooth_max_value_and_grad(h, beta=80.0):
    """
    Smooth max value and gradient w.r.t. h using log-sum-exp weights.
    
    val = abs(2N * lse / s^2) where lse = logsumexp(beta * (conv - max_conv)) / beta + max_conv
    
    We compute the gradient of the inner function f = 2N * lse / s^2, then apply sign.
    """
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf, np.zeros(N)
    conv = _fft_convolve(h)
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    weights = np.exp(scaled)
    w_sum = np.sum(weights)
    if w_sum == 0:
        weights = np.ones_like(conv) / len(conv)
    else:
        weights /= w_sum
    log_sum_exp = m + np.log(w_sum) / beta
    inner = 2.0 * N * log_sum_exp / (s * s)
    val = abs(inner)
    
    # Gradient of log_sum_exp w.r.t. h[i]:
    # lse = max_conv + (1/beta) * log(sum_k exp(beta*(conv[k]-max_conv)))
    # d(lse)/d(conv[k]) = weights[k]
    # d(conv[k])/d(h[i]) = h[i+k] (if i+k<N) + h[i-k] (if i-k>=0)
    # So d(lse)/d(h[i]) = sum_k weights[k] * (h[i+k] + h[i-k]) with bounds
    
    # Use FFT for efficiency:
    # Part 1: sum_k weights[k] * h[i+k] for i+k < N
    # Part 2: sum_k weights[k] * h[i-k] for i-k >= 0
    n_conv = len(conv)  # 2N-1
    size = 2 * n_conv
    W = np.fft.rfft(weights, size)
    H = np.fft.rfft(h, size)
    part1 = np.fft.irfft(W * H, size)[:N]
    W_flip = np.fft.rfft(weights[::-1], size)
    part2 = np.fft.irfft(W_flip * H, size)[:N]
    grad_lse = part1 + part2
    
    # d(inner)/d(h[i]) = 2N * grad_lse[i] / s^2 - 4N * lse / s^3
    grad_inner = 2.0 * N * grad_lse / (s * s) - 4.0 * N * log_sum_exp / (s ** 3)
    
    # Apply abs: if inner < 0, flip gradient
    if inner < 0:
        grad_val = -grad_inner
    else:
        grad_val = grad_inner
    
    return val, grad_val

def _full_dim_grad_refine(h0, max_iter=300, beta_start=400.0, beta_end=20.0, tol=1e-12):
    """
    Full-dimensional gradient descent with adaptive beta for the smooth surrogate.
    Uses analytic gradient of log-sum-exp smoothed max of autoconvolution.
    """
    h = h0.copy()
    best_v = _value(h)
    best_h = h.copy()
    n_iter = max_iter
    
    for it in range(n_iter):
        # Adaptive beta: start high (smoother), decrease toward exact
        t = it / max(n_iter - 1, 1)
        beta = beta_start * (beta_end / beta_start) ** t
        
        v, grad = _smooth_max_value_and_grad(h, beta)
        if not np.isfinite(v):
            break
        
        # Gradient descent with adaptive step size
        g_norm = np.linalg.norm(grad)
        if g_norm < tol:
            break
        
        # Line search: try several step sizes
        best_step_v = v
        best_step_h = h.copy()
        lr = 0.005
        for _ in range(15):
            h_trial = h - lr * grad
            v_trial = _smooth_max_value(h_trial, beta)
            if v_trial < best_step_v:
                best_step_v = v_trial
                best_step_h = h_trial.copy()
            lr *= 0.7
        
        h = best_step_h
        v_exact = _value(h)
        if v_exact < best_v:
            best_v = v_exact
            best_h = h.copy()
        
        # Deadline check
        if (time.time() - _t0) > _DEADLINE:
            break
    
    return best_h, best_v

def _dct_basis_matrix(K, N):
    """Build DCT-II basis matrix of shape (N, K) for first K modes."""
    n = np.arange(N)
    k = np.arange(K)
    angles = np.pi * (2 * n[:, None] + 1) * k[None, :] / (2 * N)
    basis = np.cos(angles)
    norms = np.sqrt(np.sum(basis ** 2, axis=0))
    norms[norms == 0] = 1.0
    basis = basis / norms
    return basis

_K_DCT = 30
_DCT_BASIS = _dct_basis_matrix(_K_DCT, N)

def _dct_h(coeffs):
    """Generate h from DCT coefficients."""
    return _DCT_BASIS @ coeffs

def _random_init(rng, n_active=None):
    """Generate a random initial h."""
    if n_active is None:
        n_active = rng.integers(5, 15)
    coeffs = np.zeros(_K_DCT)
    active_idx = rng.choice(_K_DCT, size=n_active, replace=False)
    coeffs[active_idx] = rng.normal(0, 1.0, size=n_active)
    h = _dct_h(coeffs)
    # Add small random perturbation in full dimension
    h += rng.normal(0, 0.1, size=N)
    return h

def _chirp_init(rng):
    """Generate a chirp-based initial h."""
    phase = rng.uniform(-np.pi, np.pi)
    rate = rng.uniform(-50, 50)
    amp = rng.uniform(0.5, 2.0)
    base = rng.uniform(-0.5, 0.5)
    i = _i_arr
    ang = phase + rate * (i / N) ** 2
    h = base + amp * np.cos(ang)
    return h

def construct():
    global _t0
    _t0 = time.time()
    
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    def update_best(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return best_v
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
        return v
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Multiple restarts with gradient refinement
    n_restarts = 15
    for restart in range(n_restarts):
        if not deadline_ok():
            break
        
        # Choose initialization strategy
        if restart % 3 == 0:
            h0 = _random_init(rng)
        elif restart % 3 == 1:
            h0 = _chirp_init(rng)
        else:
            h0 = _random_init(rng, n_active=rng.integers(3, 10))
        
        # Gradient refinement with adaptive beta
        h_gr, v_gr = _full_dim_grad_refine(
            h0, 
            max_iter=200, 
            beta_start=300.0, 
            beta_end=30.0,
            tol=1e-11
        )
        update_best(h_gr)
        
        # Check deadline more frequently
        if (time.time() - _t0) > _DEADLINE - 30:
            break
    
    # Phase 2: Refine the best solution found so far with tighter settings
    if deadline_ok():
        h_gr2, v_gr2 = _full_dim_grad_refine(
            best_h, 
            max_iter=200, 
            beta_start=100.0, 
            beta_end=15.0,
            tol=1e-12
        )
        update_best(h_gr2)
    
    # Phase 3: Final polish with very tight beta
    if deadline_ok():
        h_gr3, v_gr3 = _full_dim_grad_refine(
            best_h, 
            max_iter=150, 
            beta_start=50.0, 
            beta_end=10.0,
            tol=1e-13
        )
        update_best(h_gr3)
    
    # Phase 4: One more restart from a different random seed if time permits
    if deadline_ok() and (time.time() - _t0) < _DEADLINE - 60:
        rng2 = np.random.default_rng(123)
        for restart in range(5):
            if not deadline_ok():
                break
            h0 = _random_init(rng2)
            h_gr, v_gr = _full_dim_grad_refine(
                h0, 
                max_iter=150, 
                beta_start=200.0, 
                beta_end=25.0,
                tol=1e-11
            )
            update_best(h_gr)
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
