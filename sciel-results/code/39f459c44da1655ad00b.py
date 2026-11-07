import numpy as np
import time

_START_TIME = time.time()
_basis_cache = None
_x_cache = None

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF basis with 10 evenly-spaced centers and fixed widths.
    
    Strategy:
    - 10 Gaussian RBFs with centers evenly spaced across [0.1, 6.0].
    - Fixed width (sigma) chosen to provide local coverage.
    - Parameters act as independent local weights.
    - This local basis can capture non-smooth features better than global polynomials.
    - Well-conditioned for BFGS optimization.
    """
    global _basis_cache, _x_cache
    
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    p = np.asarray(params, dtype=np.float64)
    if p.ndim != 1 or p.shape[0] != 10:
        return np.zeros_like(x)

    # Time limit guard
    if time.time() - _START_TIME > 145:
        return np.zeros_like(x)

    # Check cache
    if _x_cache is not None and np.array_equal(x, _x_cache):
        basis = _basis_cache
    else:
        N = len(x)
        basis = np.zeros((10, N), dtype=np.float64)
        
        # 10 evenly spaced centers in [0.1, 6.0]
        x_min = 0.1
        x_max = 6.0
        centers = np.linspace(x_min, x_max, 10)
        
        # Fixed width: use spacing between centers as sigma
        # This gives good local coverage without too much overlap
        spacing = (x_max - x_min) / 9.0
        sigma = spacing * 0.5  # Half the spacing for moderate overlap
        
        for i in range(10):
            diff = x - centers[i]
            basis[i] = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))
        
        _x_cache = x.copy()
        _basis_cache = basis

    # Prediction: y = sum(p_i * phi_i(x))
    result = basis.T @ p
    
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result
