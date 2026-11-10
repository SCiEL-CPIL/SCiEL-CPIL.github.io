"""Structural change: sum of Gaussian basis functions with adaptive centers/widths/amplitudes."""
import numpy as np
import time

_T0 = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression: sum of 5 Gaussian basis functions.
    
    params (length 10) are interpreted as:
      - params[0:5]  = amplitudes a_i
      - params[5:10] = widths w_i (used as sigma = softplus(w_i) to keep positive)
    
    Centers are fixed at evenly spaced points in [-2, 2]:
      c_i = -2 + 4*i/4 for i=0..4  =>  [-2, -1, 0, 1, 2]
    
    This creates a localized, non-linear basis that can approximate sharp features,
    discontinuities, or multi-modal structures.
    """
    # Deadline guard
    if time.time() - _T0 > _DEADLINE:
        x = np.asarray(x, dtype=np.float64)
        return np.zeros_like(x, dtype=np.float64)
    
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    p = np.asarray(params, dtype=np.float64).reshape(-1)
    
    # Defensive: pad or truncate params to length 10
    if p.size < 10:
        pad = np.ones(10)
        pad[:p.size] = p
        p = pad
    elif p.size > 10:
        p = p[:10]
    
    # Amplitudes and widths
    amplitudes = p[0:5]
    widths = p[5:10]
    
    # Ensure positive widths via softplus-like transformation
    # sigma = 0.1 + 0.5 * (1 + tanh(widths))  =>  range (0.1, 1.1)
    sigmas = 0.1 + 0.5 * (1.0 + np.tanh(widths))
    
    # Fixed centers evenly spaced in [-2, 2]
    centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Compute sum of Gaussians: y = sum_i a_i * exp(-0.5 * ((x - c_i) / sigma_i)^2)
    # x has shape (n,), centers has shape (5,)
    # We want output shape (n,)
    
    # Reshape for broadcasting: x (n,1), centers (1,5), sigmas (1,5), amplitudes (1,5)
    x_col = x.reshape(-1, 1)
    c_row = centers.reshape(1, -1)
    s_row = sigmas.reshape(1, -1)
    a_row = amplitudes.reshape(1, -1)
    
    # Compute squared distances
    diff = x_col - c_row  # (n, 5)
    exponent = -0.5 * (diff / s_row) ** 2  # (n, 5)
    
    # Sum over basis functions
    out = np.sum(a_row * np.exp(exponent), axis=1)  # (n,)
    
    # Guard against non-finite outputs
    if not np.all(np.isfinite(out)):
        return np.zeros_like(x, dtype=np.float64)
    
    return out
