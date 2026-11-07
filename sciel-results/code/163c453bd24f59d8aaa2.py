"""Symbolic regression with a rich 10-parameter basis."""
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict y using a linear combination of 10 basis functions.
    
    The basis is designed to capture:
    - Polynomials up to degree 5 (via monomials)
    - Periodic behavior (sin, cos)
    - Exponential behavior (exp(x), exp(-x))
    - Saturation (tanh)
    - Gaussian-like bumps (exp(-x^2))
    
    This gives the BFGS fitter maximum flexibility to match common hidden functions.
    """
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Ensure params is exactly length 10
    if params.shape[0] != 10:
        params = params[:10] if params.shape[0] > 10 else np.pad(params, (0, 10 - params.shape[0]))
    
    # Compute basis functions
    # 0: constant 1
    # 1: x
    # 2: x^2
    # 3: x^3
    # 4: x^4
    # 5: x^5
    # 6: sin(x)
    # 7: cos(x)
    # 8: tanh(x)
    # 9: exp(-x^2/2)  (Gaussian bump, common in many functions)
    
    basis = np.empty((x.shape[0], 10), dtype=float)
    basis[:, 0] = 1.0
    basis[:, 1] = x
    basis[:, 2] = x * x
    basis[:, 3] = basis[:, 2] * x
    basis[:, 4] = basis[:, 2] * basis[:, 2]
    basis[:, 5] = basis[:, 4] * x
    basis[:, 6] = np.sin(x)
    basis[:, 7] = np.cos(x)
    basis[:, 8] = np.tanh(x)
    basis[:, 9] = np.exp(-0.5 * x * x)
    
    return basis @ params
