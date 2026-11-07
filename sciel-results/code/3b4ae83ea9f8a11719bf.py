"""
Strategy: Rich Linear Basis with Exponential and Hyperbolic Terms.
The model is:
f(x) = sum_{i=0}^{9} c_i * phi_i(x)

Where phi_i(x) are:
- phi_0 = 1
- phi_1 = x
- phi_2 = x^2
- phi_3 = x^3
- phi_4 = x^4
- phi_5 = sin(pi*x)
- phi_6 = cos(pi*x)
- phi_7 = tanh(x)
- phi_8 = exp(x)
- phi_9 = exp(-x)

This basis covers polynomials up to degree 4, low-frequency periodicity,
saturation (tanh), and exponential growth/decay. This should capture a wider
range of hidden functions than the previous polynomial+trig basis.
"""
import numpy as np

def _build_basis(x: np.ndarray) -> np.ndarray:
    """Build the 10-column basis matrix for input x."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    phi = np.empty((n, 10), dtype=float)
    phi[:, 0] = 1.0
    phi[:, 1] = x
    phi[:, 2] = x ** 2
    phi[:, 3] = x ** 3
    phi[:, 4] = x ** 4
    phi[:, 5] = np.sin(np.pi * x)
    phi[:, 6] = np.cos(np.pi * x)
    phi[:, 7] = np.tanh(x)
    phi[:, 8] = np.exp(x)
    phi[:, 9] = np.exp(-x)
    return phi

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    params = np.asarray(params, dtype=float).ravel()
    
    # Ensure params is length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
    
    phi = _build_basis(x)
    result = phi @ params
    result = np.where(np.isfinite(result), result, 0.0)
    return result
