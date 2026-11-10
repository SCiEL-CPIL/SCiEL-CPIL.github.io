"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

# Precompute a smooth feature matrix at the (fixed) input locations.
# We use a rich basis that covers Chebyshev polynomials (well-conditioned),
# trigonometric functions at two frequencies, and a Gaussian.
# The basis is linear in params so BFGS sees a quadratic objective and converges reliably.
_F = None
_FX = None


def _build(x):
    global _F, _FX
    _FX = x
    n = x.size
    F = np.empty((n, 10), dtype=np.float64)
    
    # Map x from [-2, 2] to u in [-1, 1] for better scaling/conditioning
    u = x / 2.0
    
    # Chebyshev polynomials T0..T4
    # T0 = 1
    F[:, 0] = 1.0
    # T1 = u
    F[:, 1] = u
    # T2 = 2u^2 - 1
    F[:, 2] = 2.0 * u * u - 1.0
    # T3 = 4u^3 - 3u
    F[:, 3] = 4.0 * u * u * u - 3.0 * u
    # T4 = 8u^4 - 8u^2 + 1
    F[:, 4] = 8.0 * u * u * u * u - 8.0 * u * u + 1.0
    
    # Trigonometric basis
    # Frequency 1 (half period over [-1,1] is pi*u, full period is 2*pi*u? 
    # Let's use pi*u and 2*pi*u to cover variations)
    # sin(pi*u), cos(pi*u)
    F[:, 5] = np.sin(np.pi * u)
    F[:, 6] = np.cos(np.pi * u)
    
    # Frequency 2
    F[:, 7] = np.sin(2.0 * np.pi * u)
    F[:, 8] = np.cos(2.0 * np.pi * u)
    
    # Gaussian centered at 0 (to capture local non-polynomial behavior)
    F[:, 9] = np.exp(-2.0 * u * u)
    
    _F = F


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    global _F, _FX
    
    # Deadline guard
    if time.time() - _T0 > _DEADLINE:
        # Return a reasonable fallback
        x = np.asarray(x, dtype=np.float64)
        return np.zeros_like(x, dtype=np.float64)
    
    x = np.asarray(x, dtype=np.float64)
    
    # (Re)build the feature matrix only if the inputs changed.
    if _F is None or _F.shape[0] != x.shape[0] or not np.allclose(_FX, x, equal_nan=True):
        _build(x)
    
    # Linear combination -> one prediction per point.
    p = np.asarray(params, dtype=np.float64).reshape(-1)
    if p.size < _F.shape[1]:
        # Defensive: pad with zeros so we never index out of range.
        pad = np.zeros(_F.shape[1])
        pad[:p.size] = p
        p = pad
    elif p.size > _F.shape[1]:
        p = p[:_F.shape[1]]
    
    out = _F @ p
    
    # Guard against non-finite outputs (would cost 1e30 / invalidate the fit).
    if not np.all(np.isfinite(out)):
        return np.zeros_like(x, dtype=np.float64)
    
    return out
# EVOLVE-BLOCK-END
