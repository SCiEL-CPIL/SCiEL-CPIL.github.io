"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(x):
    """Compute the objective: max |autoconv| / (sum x)^2.
    x is a 1D array of step heights."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    # Autoconvolution via FFT for speed
    n_conv = 2 * N - 1
    fft_size = 1
    while fft_size < n_conv:
        fft_size *= 2
    X = np.fft.rfft(x, n=fft_size)
    ac = np.fft.irfft(X * X, n=fft_size)[:n_conv]
    max_abs = np.max(np.abs(ac))
    return max_abs / (s * s)

def _gradient(x):
    """Approximate gradient of the objective using finite differences or analytical."""
    # For efficiency, we'll use a simple numerical gradient with small step
    eps = 1e-7
    grad = np.zeros_like(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return grad
    # Compute autoconvolution once
    n_conv = 2 * N - 1
    fft_size = 1
    while fft_size < n_conv:
        fft_size *= 2
    X = np.fft.rfft(x, n=fft_size)
    ac = np.fft.irfft(X * X, n=fft_size)[:n_conv]
    
    # The objective is max_k |ac_k| / s^2
    # This is complex to differentiate analytically quickly.
    # Let's use a simpler approach: random restarts + local search with gradient-free or simple gradient.
    # Actually, let's try a different strategy: use scipy minimize on a smooth surrogate.
    return grad  # Placeholder, we'll use derivative-free or a different method

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_x = None
    best_score = np.inf
    
    # Strategy: 
    # 1. Try several random initializations
    # 2. For each, do local optimization using scipy (Nelder-Mead or Powell)
    # 3. Track the best
    
    # We want to minimize max|autoconv| / (sum x)^2
    # Note: scaling x by c scales numerator by c^2 and denominator by c^2, so score is scale-invariant.
    # We can fix sum(x) = 1 to remove this degree of freedom.
    # Let x = y / sum(y), then score = max|autoconv(y)| / (sum(y))^2 = max|autoconv(y/sum(y))| / 1
    # Actually, let's just work with normalized variables.
    
    # Better: parameterize x such that sum(x) = 1.
    # Let x = softmax(z) or x = (1 + z) / sum(1+z) with z free.
    # Simpler: use x directly but normalize before scoring.
    
    # Let's define a function that takes z and returns normalized x and score
    def score_from_z(z):
        # Use tanh to keep values bounded, then normalize
        x = np.tanh(z)
        s = np.sum(x)
        if abs(s) < 1e-10:
            return np.inf
        x_norm = x / s
        return _compute_score(x_norm)
    
    # Try multiple random starts
    n_starts = 20
    rng = np.random.default_rng(42)
    
    for i in range(n_starts):
        if time.time() > deadline:
            break
        
        # Random initialization
        z0 = rng.standard_normal(N) * 0.1
        
        # Use Nelder-Mead for derivative-free optimization
        try:
            result = minimize(score_from_z, z0, method='Nelder-Mead', 
                             options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if result.success or result.fun < best_score:
                z_opt = result.x
                x_opt = np.tanh(z_opt)
                s = np.sum(x_opt)
                if abs(s) > 1e-10:
                    x_opt = x_opt / s
                    sc = _compute_score(x_opt)
                    if sc < best_score:
                        best_score = sc
                        best_x = x_opt.copy()
        except Exception:
            continue
    
    # If we still have time, do a final polish with more iterations
    if best_x is not None and time.time() < deadline - 5.0:
        # Refine the best solution
        z_best = np.arctanh(np.clip(best_x, -0.999, 0.999))
        try:
            result = minimize(score_from_z, z_best, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_score:
                z_opt = result.x
                x_opt = np.tanh(z_opt)
                s = np.sum(x_opt)
                if abs(s) > 1e-10:
                    x_opt = x_opt / s
                    best_x = x_opt.copy()
        except Exception:
            pass
    
    if best_x is None:
        # Fallback: return ones
        return np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
