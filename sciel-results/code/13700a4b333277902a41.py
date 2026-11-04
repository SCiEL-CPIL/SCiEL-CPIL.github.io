"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_value(h):
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _soft_max_surrogate(h, beta=50.0):
    s = np.sum(h)
    if s == 0:
        return float('inf')
    conv = np.convolve(h, h)
    # Soft max to make differentiable
    scaled = beta * conv / (N * N)
    m = np.max(scaled)
    logsumexp = m + np.log(np.sum(np.exp(scaled - m)))
    return abs(2.0 * N * (logsumexp / beta) / (s * s))


def _objective(x):
    h = np.tanh(x)
    return _soft_max_surrogate(h)


def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill
    
    # Start from a low-frequency sinusoidal initialization
    # This gives a smoother autoconvolution peak
    i = np.arange(N)
    h_init = 1.0 + 0.5 * np.cos(2.0 * np.pi * i / N)
    # Ensure positive to use tanh parametrization well
    h_init = h_init - np.min(h_init) + 0.1
    
    best_h = h_init.copy()
    best_val = _compute_value(best_h)
    
    # Try L-BFGS-B optimization
    x_init = np.arctanh(np.clip(h_init, -0.99, 0.99))
    
    try:
        result = minimize(
            _objective,
            x_init,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000}
        )
        
        if time.time() < deadline:
            h_opt = np.tanh(result.x)
            val_opt = _compute_value(h_opt)
            if val_opt < best_val:
                best_h = h_opt
                best_val = val_opt
    except Exception:
        pass
    
    # Try a few different sinusoidal initializations if time permits
    if time.time() < deadline - 30:
        for freq in [2, 3, 5]:
            if time.time() >= deadline - 20:
                break
            h_try = 1.0 + 0.3 * np.cos(2.0 * np.pi * freq * i / N)
            h_try = h_try - np.min(h_try) + 0.1
            x_try = np.arctanh(np.clip(h_try, -0.99, 0.99))
            try:
                result = minimize(
                    _objective,
                    x_try,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 1000}
                )
                if time.time() < deadline:
                    h_opt = np.tanh(result.x)
                    val_opt = _compute_value(h_opt)
                    if val_opt < best_val:
                        best_h = h_opt
                        best_val = val_opt
            except Exception:
                pass
    
    return best_h
# EVOLVE-BLOCK-END
