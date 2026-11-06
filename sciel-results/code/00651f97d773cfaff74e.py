import time
import numpy as np
from scipy.optimize import minimize

N = 400
_t0 = time.time()
_DEADLINE = 135.0

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _objective_and_grad(h, beta=100.0):
    """
    Compute objective value and gradient for smoothed LSE approximation.
    Objective: V = 2 * N * LSE(conv) / S^2
    """
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf, np.zeros_like(h)
    
    conv = np.convolve(h, h)
    m = np.max(conv)
    
    # Numerical stability
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)
    
    if sum_exp == 0:
        lse = m
        p = np.zeros_like(conv)
    else:
        lse = m + np.log(sum_exp) / beta
        p = exp_scaled / sum_exp
        
    val = abs(2.0 * N * lse / (s * s))
    
    # Gradient of LSE wrt h
    # d(LSE)/dh_j = 2 * (p * h)_j
    # Using FFT for efficiency
    size = 2 * N
    P = np.fft.rfft(p, size)
    H = np.fft.rfft(h, size)
    grad_lse_fft = np.fft.irfft(P * H, size)[:N]
    grad_lse = 2.0 * grad_lse_fft
    
    # Gradient of V = 2N * LSE / S^2
    # dV/dh = (2N/S^2) * (dLSE/dh - 2*LSE/S)
    # Note: We are minimizing abs(V). Since V is positive (max conv >= 0, S^2 > 0),
    # minimizing V is equivalent.
    # If LSE is negative? No, LSE approximates max, and max >= 0.
    # So V > 0.
    
    grad_val = (2.0 * N / (s * s)) * (grad_lse - (2.0 * lse / s))
    
    return val, grad_val

def construct():
    global _t0
    _t0 = time.time()
    
    rng = np.random.default_rng(42)
    
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    def update_best(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
            
    # Try a few random starts
    for restart in range(5):
        if not deadline_ok():
            break
            
        # Initialize
        if restart == 0:
            h0 = np.ones(N)
        else:
            h0 = rng.normal(0, 1, N)
            # Ensure sum is not zero
            if np.sum(h0) == 0:
                h0[0] += 1.0
                
        # Define objective for optimizer
        def obj(h):
            val, _ = _objective_and_grad(h)
            return val
        
        def grad(h):
            _, g = _objective_and_grad(h)
            return g
            
        try:
            res = minimize(obj, h0, jac=grad, method='L-BFGS-B', 
                           options={'maxiter': 500, 'maxfun': 1000, 'disp': False})
            if res.success or res.fun < best_v:
                update_best(res.x)
        except Exception:
            pass
            
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    
    return best_h
