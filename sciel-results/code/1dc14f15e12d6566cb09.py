import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # Safety margin before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a hybrid basis of Chebyshev polynomials and
    Gaussian RBFs.
    
    Chebyshev polynomials provide a well-conditioned global approximation
    for smooth functions on bounded intervals. Gaussian RBFs provide local
    flexibility to capture sharp features or non-smooth behavior.
    
    We allocate 6 parameters to Chebyshev (degrees 0-5) and 4 parameters
    to RBFs with fixed centers and widths. This balances global trend
    capture with local adaptability.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a simple linear fit as fallback
        return params[0] + params[1] * x
    
    N = x.shape[0]
    
    # Map x from [-2, 2] to t in [-1, 1] for Chebyshev
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_5 (6 terms)
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2*t*T_n(t) - T_{n-1}(t)
    
    # Build basis matrix: (N, 10)
    basis = np.zeros((N, 10), dtype=np.float64)
    
    # Chebyshev basis (columns 0-5)
    basis[:, 0] = 1.0  # T_0
    basis[:, 1] = t    # T_1
    
    for n in range(2, 6):
        basis[:, n] = 2.0 * t * basis[:, n-1] - basis[:, n-2]
    
    # Gaussian RBF basis (columns 6-9)
    # Fixed centers spread across [-2, 2]
    centers = np.array([-1.0, -0.5, 0.5, 1.0])
    # Fixed widths: moderate scale for local flexibility
    widths = np.array([0.5, 0.5, 0.5, 0.5])
    
    for k in range(4):
        diff = x - centers[k]
        basis[:, 6 + k] = np.exp(-diff**2 / (2.0 * widths[k]**2))
    
    # Linear combination
    result = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
    
    return result
