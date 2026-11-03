"""Rewritten program: Subgradient/Smooth optimization for low off-diagonal autocorrelation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv(x):
    """Linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    X = np.fft.rfft(x, fsize)
    c = np.fft.irfft(X * X, fsize)[:size]
    return c

def _metric_offdiag(x):
    """Compute max |c_k| for k != 0 divided by (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv(x)
    # Exclude k=0
    if len(c) > 1:
        max_val = np.max(np.abs(c[1:]))
    else:
        max_val = 0.0
    return max_val / (s * s)

def _objective_smooth(x, temp=1.0):
    """Smooth approximation of the off-diagonal metric for optimization."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    c = _autoconv(x)
    if len(c) > 1:
        abs_c = np.abs(c[1:])
    else:
        return 0.0
    # Soft max: temp * log(sum(exp(c/temp)))
    max_c = np.max(abs_c)
    soft_max = max_c + temp * np.log(np.sum(np.exp((abs_c - max_c) / temp)))
    return soft_max / (s * s)

def construct():
    t0 = time.time()
    deadline = t0 + 110.0
    
    best_x = np.ones(N)
    best_val = _metric_offdiag(best_x)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. All ones
    candidates.append(np.ones(N))
    
    # 2. Random sequences with non-zero mean
    for _ in range(10):
        r = rng.random(N)
        # Bias towards positive to ensure non-zero sum
        x = np.where(r > 0.4, 1.0, -1.0)
        if abs(np.sum(x)) > 10:
            candidates.append(x)
            
    # 3. M-sequences (if available, otherwise approximate)
    # Simple pseudo-random binary sequence
    for _ in range(5):
        x = rng.choice([-1, 1], N)
        if abs(np.sum(x)) > 10:
            candidates.append(x)

    # 4. Structured: Block patterns
    for block_size in [10, 20, 50]:
        pattern = np.zeros(N)
        for i in range(N):
            if (i // block_size) % 2 == 0:
                pattern[i] = 1.0
            else:
                pattern[i] = -1.0
        # Add small bias to ensure sum != 0
        pattern[0] += 0.1
        candidates.append(pattern)

    # Optimize top candidates
    for x0 in candidates:
        if time.time() > deadline - 20:
            break
            
        x0 = x0.copy()
        # Normalize sum to 1 to handle scale invariance
        s0 = np.sum(x0)
        if abs(s0) < 1e-6:
            continue
        x0 = x0 / s0
        
        # Use L-BFGS-B with smooth objective
        try:
            res = minimize(
                _objective_smooth,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            x_opt = res.x
            # Restore scale? No, scale is invariant.
            # But we need to check the actual metric
            val = _metric_offdiag(x_opt)
            if val < best_val:
                best_val = val
                best_x = x_opt.copy()
        except Exception:
            pass
            
    return best_x
