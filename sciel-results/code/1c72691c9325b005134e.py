"""
Symbolic regression using a hybrid Chebyshev + RBF basis.

Design:
- 5 Chebyshev polynomials T_0..T_4 on scaled domain [-1, 1] for smooth global structure.
- 5 Radial Basis Functions (Gaussians) with centers spanning [-2, 2] for local features.
- Basis columns are normalized to unit L2 norm for well-conditioned BFGS optimization.
- BFGS fits the 10 params from all-ones start.
- Includes a deadline guard to stay within the 150s wall-clock limit.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # Safety margin before 150s limit

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x: array of input points (uniformly from [-2, 2])
    params: array of 10 coefficients
    Returns: predicted values, same shape as x
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Deadline guard (lightweight check)
    if time.time() - _start_time > _DEADLINE:
        return np.full_like(x, params[0])
    
    # Scale x from [-2, 2] to [-1, 1] for Chebyshev polynomials
    t = 0.5 * x
    
    # Chebyshev polynomials T_0 to T_4
    b0 = np.ones_like(x)
    b1 = t
    b2 = 2.0 * t**2 - 1.0
    b3 = 4.0 * t**3 - 3.0 * t
    b4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    
    # Radial Basis Functions (Gaussians)
    # Centers evenly spaced across [-2, 2] to cover the full input range
    centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    sigma = 1.0  # Width parameter, broader coverage for smoother local features
    
    # Compute Gaussians: shape (N, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    gaussians = np.exp(-0.5 * (diff / sigma)**2)
    
    # Build full basis matrix of shape (N, 10)
    # Columns 0-4: Chebyshev
    # Columns 5-9: Gaussians
    B = np.column_stack((b0, b1, b2, b3, b4, gaussians))
    
    # Normalize each column to unit L2 norm for well-conditioned optimization
    # This helps BFGS converge more reliably when columns have different scales
    col_norms = np.sqrt(np.sum(B**2, axis=0))
    col_norms[col_norms < 1e-12] = 1.0  # avoid division by zero
    B = B / col_norms[np.newaxis, :]
    
    # Compute prediction
    pred = B @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred
# EVOLVE-BLOCK-END
