import numpy as np
import time

N = 400
DEADLINE = 150.0
SAFETY_MARGIN = 10.0

def _autoconv(x):
    """Compute full linear autoconvolution of x (length N) -> length 2N-1."""
    n = 2 * N
    X = np.fft.rfft(x, n=n)
    C = np.fft.irfft(X * X, n=n).real[:2 * N - 1]
    return C

def _eval_ratio(x):
    """Compute the raw value: max|c_k| / (sum x)^2."""
    c = _autoconv(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    return np.max(np.abs(c)) / (s * s)

def _soft_max_abs(c, beta):
    """Smooth surrogate for max|c_k| using log-sum-exp."""
    ac = np.abs(c)
    m = np.max(ac)
    if m < 1e-300:
        return 0.0
    ac = ac / m
    z = np.log(np.sum(np.exp(beta * ac))) / beta + m
    return z

def _smooth_metric(x, beta):
    """Differentiable proxy for _eval_ratio."""
    c = _autoconv(x)
    s = np.sum(x)
    if abs(s) < 1e-10:
        s = 1e-10 * (1.0 if s >= 0 else -1.0)
    smax = _soft_max_abs(c, beta)
    return smax / (s * s)

def _gradient_smooth(x, beta):
    """Compute gradient of smooth metric w.r.t. x using FFT."""
    n = 2 * N
    X = np.fft.rfft(x, n=n)
    # C_k = sum_j x_j x_{j+k} for linear conv, index k
    # In frequency: C = IFFT(|X|^2) but for linear we pad
    # Actually autoconv via FFT: C = ifft(X * X) gives circular conv of padded x
    # For linear conv of x with x, we pad to length >= 2N-1
    # Let's use the standard: full = ifft(X * X) where X = fft(x, n)
    # But we need d/dx of max|C|/(sum x)^2
    # Let's just do numerical gradient for simplicity, or use FFT chain rule
    
    # Actually let's compute gradient numerically for reliability
    grad = np.zeros(N)
    eps = 1e-6
    base = _smooth_metric(x, beta)
    for i in range(N):
        xp = x.copy()
        xp[i] += eps
        xm = x.copy()
        xm[i] -= eps
        grad[i] = (_smooth_metric(xp, beta) - _smooth_metric(xm, beta)) / (2 * eps)
    return grad

def _gradient_descent_smooth(x0, beta, max_iter, step_size, deadline):
    """Run gradient descent with adaptive step size on smooth metric."""
    x = x0.copy()
    best_x = x.copy()
    best_val = _eval_ratio(x)
    
    # We'll optimize the smooth metric but track actual ratio
    cur_smooth = _smooth_metric(x, beta)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        grad = _gradient_smooth(x, beta)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        # Adaptive step: try a few step sizes
        best_trial = cur_smooth
        best_trial_x = x.copy()
        
        for trial_frac in [1.0, 0.5, 0.25, 0.1, 0.05]:
            trial_step = step_size * trial_frac
            x_trial = x - trial_step * grad
            trial_smooth = _smooth_metric(x_trial, beta)
            if trial_smooth < best_trial:
                best_trial = trial_smooth
                best_trial_x = x_trial.copy()
        
        x = best_trial_x
        cur_smooth = best_trial
        
        # Track actual metric
        actual = _eval_ratio(x)
        if actual < best_val:
            best_val = actual
            best_x = x.copy()
    
    return best_x, best_val

def construct():
    """
    Return signed step heights that minimize max|c_k| / (sum x_i)^2.
    
    Uses spectral-design / gradient descent on a smooth surrogate to find
    sequences with autoconvolution concentrated away from the peak, reducing
    max|c_k| relative to (sum x)^2.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    # Start with constant sequence (known lower bound 1/N for reals)
    x_const = np.ones(N)
    best_val = _eval_ratio(x_const)
    best_result = x_const.copy()
    
    # Try several random initializations with gradient descent
    np.random.seed(42)
    
    # Phase 1: Random starts with moderate beta
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Mix of random real and structured starts
        if trial % 3 == 0:
            x0 = np.random.randn(N)
        elif trial % 3 == 1:
            x0 = np.random.choice([-1.0, 1.0], N)
        else:
            # Smooth random: low-pass filtered
            x0 = np.random.randn(N)
            from scipy.ndimage import gaussian_filter
            x0 = gaussian_filter(x0, sigma=3.0)
        
        # Normalize to have sum = 1 for stability
        s0 = np.sum(x0)
        if abs(s0) < 1e-10:
            x0 += 0.1
            s0 = np.sum(x0)
        x0 = x0 / s0
        
        beta = 50.0 + 10.0 * trial  # increasing smoothness
        max_iter = 500
        step_size = 0.01
        
        x_opt, val = _gradient_descent_smooth(x0, beta, max_iter, step_size, deadline)
        if val < best_val:
            best_val = val
            best_result = x_opt.copy()
    
    # Phase 2: If time remains, try lower beta (sharper max) for polish
    if time.time() < deadline - 10.0:
        for trial in range(5):
            if time.time() > deadline:
                break
            beta = 200.0 + 100.0 * trial
            max_iter = 300
            step_size = 0.005
            
            x_opt, val = _gradient_descent_smooth(best_result, beta, max_iter, step_size, deadline)
            if val < best_val:
                best_val = val
                best_result = x_opt.copy()
    
    # Final: also check the constant sequence in case everything got worse
    val_const = _eval_ratio(x_const)
    if val_const < best_val:
        best_val = val_const
        best_result = x_const.copy()
    
    return best_result

# EVOLVE-BLOCK-END
