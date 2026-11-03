import numpy as np
from scipy.optimize import minimize
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _autoconv(x):
    """Compute full autoconvolution using FFT for efficiency."""
    n = len(x)
    # Zero-pad to avoid circular convolution
    size = 2 * n
    X = np.fft.rfft(x, size)
    ac = np.fft.irfft(X * X.conj(), size)[:n]
    # Full autoconvolution has 2n-1 lags, but irfft gives us n values (0 to n-1)
    # We need lags from -(n-1) to (n-1). The irfft result gives lags 0 to n-1.
    # For real signals, c[-k] = c[k], so max |c[k]| = max of |ac| for k=0..n-1
    return ac

def _objective(x, beta=100.0):
    """Smooth approximation of max|c[k]| / (sum x)^2 using log-sum-exp."""
    ac = _autoconv(x)
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    # We want max |ac[k]| / s^2
    # Use log-sum-exp for smooth max of |ac[k]|
    # |ac[k]| = exp(log(|ac[k]|)), but ac can be negative, so use abs
    abs_ac = np.abs(ac)
    # Add small epsilon to avoid log(0)
    abs_ac_safe = np.maximum(abs_ac, 1e-12)
    log_abs = np.log(abs_ac_safe)
    # Log-sum-exp for smooth max
    m = np.max(log_abs)
    log_sum_exp = m + np.log(np.sum(np.exp(log_abs - m)))
    # The objective is exp(log_sum_exp) / s^2
    # For optimization, we can minimize log_sum_exp - 2*log|s|
    return log_sum_exp - 2.0 * np.log(abs(s))

def _true_score(x):
    """Compute the actual score: max|ac[k]| / (sum x)^2."""
    ac = _autoconv(x)
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    return np.max(np.abs(ac)) / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initialize with a structured pattern
    # Try several initializations and pick the best
    candidates = []
    
    # Init 1: Alternating pattern with a bias to ensure sum != 0
    x1 = np.ones(N)
    x1[1::2] = -1.0
    # Add a small positive bias to make sum positive and non-zero
    # sum = 1 - (N-1)/2 ... for N=400, sum = 1 - 199 = -198. Not great.
    # Let's try: mostly +1 with some -1
    # Actually, let's use a different approach.
    
    # Init 2: Random with zero mean removed, then add constant
    rng = np.random.default_rng(42)
    x2 = rng.standard_normal(N)
    x2 -= np.mean(x2)  # zero mean
    x2 += 1.0  # shift so sum = N
    
    # Init 3: Block pattern
    x3 = np.ones(N)
    block = N // 4
    x3[block:2*block] = -1.0
    
    # Init 4: Cosine-based
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    x4 = 1.0 + 0.5 * np.cos(t)
    
    for x_init in [x1, x2, x3, x4]:
        candidates.append(x_init.copy())
    
    best_x = None
    best_score = float('inf')
    
    # Evaluate initial candidates
    for x_c in candidates:
        s = _true_score(x_c)
        if s < best_score:
            best_score = s
            best_x = x_c.copy()
    
    # Gradient-based optimization on the best candidate
    x = best_x.copy()
    
    # Normalize so sum = 1 to simplify
    s0 = np.sum(x)
    if abs(s0) > 1e-10:
        x = x / s0
    
    # Use L-BFGS-B with bounds to keep values reasonable
    # No explicit bounds, but the objective is scale-invariant so it should be fine
    
    def obj_and_grad(x_vec):
        # Compute objective and gradient numerically or analytically
        # For simplicity, use numerical gradient via scipy
        return _objective(x_vec)
    
    # Do a few optimization passes
    n_iters = 0
    while time.time() < deadline - 10.0:
        result = minimize(
            obj_and_grad,
            x,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        n_iters += 1
        if result.success or result.status == 0:
            x_new = result.x
            s_new = _true_score(x_new)
            if s_new < best_score:
                best_score = s_new
                best_x = x_new.copy()
                x = x_new.copy()
        else:
            # Try a different beta for the smooth approximation
            break
        
        if time.time() > deadline - 15.0:
            break
    
    # Try local search: flip individual elements or adjust values
    # Simple local search: for each position, try perturbing the value
    x = best_x.copy()
    s = _true_score(x)
    
    # Normalize
    sm = np.sum(x)
    if abs(sm) > 1e-10:
        x = x / sm
    s = _true_score(x)
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Try perturbing each element
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            for delta in [-0.5, -0.1, 0.1, 0.5]:
                x_test = x.copy()
                x_test[i] += delta
                s_test = _true_score(x_test)
                if s_test < s * (1 - 1e-8):
                    x = x_test.copy()
                    s = s_test
                    improved = True
                    break
    
    # Final polish: more iterations of gradient descent
    if time.time() < deadline - 3.0:
        sm = np.sum(x)
        if abs(sm) > 1e-10:
            x = x / sm
        result = minimize(
            obj_and_grad,
            x,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        if result.success:
            s_final = _true_score(result.x)
            if s_final < _true_score(x):
                x = result.x
    
    return x
# EVOLVE-BLOCK-END
