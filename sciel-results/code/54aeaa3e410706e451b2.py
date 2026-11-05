"""Optimized program for the third autocorrelation inequality task."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_fft(h):
    """Compute full autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    # Pad to next power of 2 for FFT efficiency
    fft_size = 1 << (size - 1).bit_length()
    h_pad = np.zeros(fft_size)
    h_pad[:n] = h
    H = np.fft.fft(h_pad)
    conv = np.fft.ifft(H * H).real[:size]
    return conv

def _soft_max(x, beta=100.0):
    """Smooth approximation of max(x) using soft-max."""
    m = np.max(x)
    return m + np.log(np.sum(np.exp(beta * (x - m)))) / beta

def _value_exact(h):
    """Compute the exact objective value."""
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    return abs(2.0 * N * max_conv / (s * s))

def _sparse_to_h(params, n_spikes=10):
    """
    Convert sparse parameter vector to h array.
    params: [pos_1, amp_1, pos_2, amp_2, ..., pos_k, amp_k, offset]
    where pos_i in [0, N), amp_i in R, offset is added to all.
    """
    n = N
    h = np.zeros(n)
    n_pairs = n_spikes
    for i in range(n_pairs):
        pos = int(params[2*i] % n)
        amp = params[2*i + 1]
        h[pos] += amp
    offset = params[2*n_pairs]
    h += offset
    return h

def _objective_sparse(params, n_spikes=10):
    """Objective for sparse parameterization using soft-max surrogate."""
    h = _sparse_to_h(params, n_spikes)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _autoconv_fft(h)
    # Use soft-max to make it differentiable
    max_conv = _soft_max(conv, beta=50.0)
    return abs(2.0 * N * max_conv / (s * s))

def _construct_optimized():
    """Optimize h to minimize the autocorrelation ratio using sparse parameterization."""
    deadline = time.time() + 130  # 130s budget, leave 20s margin
    
    best_h = np.ones(N)
    best_val = _value_exact(best_h)
    
    rng = np.random.default_rng(123)
    
    # Try different numbers of spikes
    for n_spikes in [5, 10, 20]:
        if time.time() > deadline - 30:
            break
        
        # Random initializations
        for trial in range(3):
            if time.time() > deadline - 15:
                break
            
            params = np.zeros(2 * n_spikes + 1)
            # Random spike positions and amplitudes
            for i in range(n_spikes):
                params[2*i] = rng.uniform(0, N)
                params[2*i + 1] = rng.uniform(-2.0, 2.0)
            params[2*n_spikes] = rng.uniform(-0.5, 0.5)
            
            try:
                result = minimize(
                    _objective_sparse,
                    params,
                    args=(n_spikes,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                h_candidate = _sparse_to_h(result.x, n_spikes)
                val_candidate = _value_exact(h_candidate)
                if np.isfinite(val_candidate) and val_candidate < best_val:
                    best_val = val_candidate
                    best_h = h_candidate.copy()
            except Exception:
                pass
    
    # Local refinement on best_h using full optimization
    if time.time() < deadline - 10:
        try:
            result = minimize(
                _value_exact,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            val_final = _value_exact(result.x)
            if np.isfinite(val_final) and val_final < best_val:
                best_val = val_final
                best_h = result.x.copy()
        except Exception:
            pass
    
    # Final validity check
    final_val = _value_exact(best_h)
    if not np.isfinite(final_val) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h

def construct():
    """Construct step heights minimizing the autocorrelation ratio."""
    try:
        return _construct_optimized()
    except Exception:
        return np.ones(N)
# EVOLVE-BLOCK-END
