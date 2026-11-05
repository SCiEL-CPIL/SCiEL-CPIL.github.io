"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _zadoff_chu(N, root):
    """Generate a Zadoff-Chu sequence of length N with given root."""
    n = np.arange(N)
    # Zadoff-Chu: exp(-j*pi*root*n*(n+1)/N)
    phase = -np.pi * root * n * (n + 1) / N
    return np.exp(1j * phase)

def _hann_window(N):
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))

def _softmax_max(x, alpha=50.0):
    """Soft approximation of max(x) using log-sum-exp."""
    m = np.max(x)
    if m < -1e10:
        return m
    # Use stable log-sum-exp
    x_shifted = x - m
    return m + np.log(np.sum(np.exp(alpha * x_shifted))) / alpha

def _objective(params, N, alpha=50.0):
    """Objective: minimize soft-max of autoconvolution / sum^2."""
    h = params
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    # We want to minimize max(conv) / s^2
    # Use soft max for differentiability
    soft_max = _softmax_max(conv, alpha)
    return soft_max / (s * s)

def _value(h):
    """Compute the actual task value."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2 * N * np.max(conv) / (s * s))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Initial candidate: windowed Zadoff-Chu with chirp, real part
    best_h = np.ones(N)
    best_val = _value(best_h)
    
    # Try several initial constructions
    candidates = []
    
    # Zadoff-Chu with various roots, windowed
    for root in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        zc = _zadoff_chu(N, root)
        # Apply Hann window
        w = _hann_window(N)
        # Chirp modulation: multiply by exp(j*pi*n^2/N)
        n = np.arange(N)
        chirp = np.exp(1j * np.pi * n**2 / N)
        h_cand = np.real(zc * chirp * w)
        # Ensure non-zero sum
        if abs(np.sum(h_cand)) > 1e-10:
            candidates.append(h_cand)
    
    # Also try simple windowed sequences
    for w_type in ['hann', 'hamming', 'blackman']:
        if w_type == 'hann':
            w = _hann_window(N)
        elif w_type == 'hamming':
            w = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))
        else:
            w = np.blackman(N)
        # Add some structure
        n = np.arange(N)
        h_cand = w * np.cos(2 * np.pi * n / N)
        if abs(np.sum(h_cand)) > 1e-10:
            candidates.append(h_cand)
    
    # Evaluate all candidates
    for h_cand in candidates:
        val = _value(h_cand)
        if val < best_val:
            best_val = val
            best_h = h_cand.copy()
    
    # Local optimization on the best candidate
    if time.time() < deadline - 30:
        h_opt = best_h.copy()
        alpha_values = [10.0, 20.0, 50.0]
        
        for alpha in alpha_values:
            if time.time() > deadline - 20:
                break
            try:
                result = minimize(
                    _objective, h_opt, args=(N, alpha),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.success or result.fun < _objective(h_opt, N, alpha):
                    h_candidate = result.x
                    val = _value(h_candidate)
                    if val < best_val:
                        best_val = val
                        best_h = h_candidate.copy()
            except Exception:
                pass
    
    # Final high-precision polish with Nelder-Mead on a smaller neighborhood
    if time.time() < deadline - 15:
        try:
            # Use the current best as starting point
            result = minimize(
                _objective, best_h, args=(N, 100.0),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            h_candidate = result.x
            val = _value(h_candidate)
            if val < best_val:
                best_val = val
                best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Ensure validity
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-10:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
