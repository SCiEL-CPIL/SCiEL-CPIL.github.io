"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

# Precompute a smooth feature matrix at the (fixed) input locations.
# We use a rich basis that covers polynomials, trigonometric, Gaussian, and
# sigmoid-like functions. The basis is linear in params so BFGS sees a
# quadratic objective and converges reliably from the all-ones start.
_F = None
_FX = None


def _build(x):
    global _F, _FX
    _FX = x
    n = x.size
    F = np.empty((n, 10), dtype=np.float64)
    
    # Polynomials up to degree 4
    F[:, 0] = 1.0
    F[:, 1] = x
    F[:, 2] = x * x
    F[:, 3] = x * x * x
    F[:, 4] = x * x * x * x
    
    # Trigonometric
    F[:, 5] = np.sin(x)
    F[:, 6] = np.cos(x)
    
    # Gaussian
    F[:, 7] = np.exp(-x * x)
    
    # Sigmoid-like
    F[:, 8] = np.tanh(x)
    
    # Odd Gaussian-like
    F[:, 9] = x * np.exp(-x * x)
    
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
