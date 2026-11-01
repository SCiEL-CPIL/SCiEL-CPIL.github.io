"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_start_time = time.time()
_DEADLINE = 140  # seconds, safety margin before 150s kill

def _autoconv_max_abs(s):
    """Compute max absolute value of autoconvolution of s with itself."""
    n = len(s)
    # Autoconvolution: c[k] = sum_{i} s[i] * s[i+k] for k=0..n-1
    # Use FFT for efficiency
    f = np.fft.fft(s, n=2*n)
    conv = np.fft.ifft(f * f).real[:n]
    return np.max(np.abs(conv))

def _objective(x):
    """Objective: max absolute autoconvolution, mapped from continuous x in [-1,1]."""
    s = x  # continuous relaxation
    n = len(s)
    f = np.fft.fft(s, n=2*n)
    conv = np.fft.ifft(f * f).real[:n]
    return np.max(np.abs(conv))

def _gradient(x):
    """Approximate gradient via finite differences (sparse)."""
    n = len(x)
    grad = np.zeros(n)
    eps = 1e-6
    f0 = _objective(x)
    # Only perturb a subset to save time
    idxs = np.random.choice(n, size=min(50, n), replace=False)
    for i in idxs:
        x2 = x.copy()
        x2[i] += eps
        grad[i] = (_objective(x2) - f0) / eps
    return grad

def construct():
    global _start_time
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        return np.ones(N)
    
    # Start with all ones (seed)
    best_s = np.ones(N)
    best_val = _autoconv_max_abs(best_s)
    
    # Try gradient-based optimization on continuous relaxation
    # Multiple random restarts
    n_restarts = 3
    for r in range(n_restarts):
        if time.time() - _start_time > _DEADLINE - 5:
            break
        
        # Initialize: mix of +1 and -1 with some randomness
        if r == 0:
            x0 = np.ones(N)
        else:
            x0 = np.random.choice([-1.0, 1.0], size=N).astype(float)
        
        # Add small noise to escape symmetry
        x0 = x0 + np.random.randn(N) * 0.1
        x0 = np.clip(x0, -1.0, 1.0)
        
        # Use L-BFGS-B
        try:
            res = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                bounds=[(-1, 1)] * N,
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            # Round to ±1
            s_rounded = np.sign(res.x)
            s_rounded[s_rounded == 0] = 1
            
            val = _autoconv_max_abs(s_rounded)
            if val < best_val:
                best_val = val
                best_s = s_rounded.copy()
                
        except Exception:
            continue
    
    # Local search: flip individual bits to improve
    if time.time() - _start_time > _DEADLINE - 3:
        return best_s
    
    # Greedy local search
    improved = True
    max_iters = 200
    iters = 0
    while improved and iters < max_iters:
        if time.time() - _start_time > _DEADLINE - 1:
            break
        improved = False
        iters += 1
        # Try flipping each position
        for i in range(N):
            if time.time() - _start_time > _DEADLINE - 1:
                break
            old = best_s[i]
            best_s[i] = -old
            val = _autoconv_max_abs(best_s)
            if val < best_val:
                best_val = val
                improved = True
            else:
                best_s[i] = old
    
    return best_s
# EVOLVE-BLOCK-END
