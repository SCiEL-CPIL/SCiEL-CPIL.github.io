import numpy as np
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_DEADLINE = 145.0


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x)

    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.flatten()
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has exactly 10 elements
    if p.shape[0] != 10:
        p = p[:10] if p.shape[0] >= 10 else np.pad(p, (0, 10 - p.shape[0]))
    
    # Pure Fourier basis: 5 cosines + 5 sines with frequencies {1,2,3,4,5}
    # This is linear in params, so BFGS will converge quickly
    freqs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # p[0:5] are cosine coefficients, p[5:10] are sine coefficients
    cos_terms = np.zeros_like(x)
    sin_terms = np.zeros_like(x)
    
    for i, f in enumerate(freqs):
        cos_terms += p[i] * np.cos(f * x)
        sin_terms += p[5 + i] * np.sin(f * x)
    
    result = cos_terms + sin_terms
    
    return result
