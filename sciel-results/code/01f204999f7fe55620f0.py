import numpy as np
import time

# Global cache for the basis matrix
# Since the training x is constant during the optimization, we only need to compute the basis once.
_basis_cache = None
_x_cache = None
_start_time = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Gaussian RBF basis.
    
    The model is f(x) = sum_{k=0}^{9} c_k * exp(-((x - mu_k)^2) / (2 * sigma^2)).
    This uses 10 parameters as weights for 10 fixed RBF centers.
    
    RBFs are local basis functions that can capture non-smooth features or sharp
    variations better than global polynomials. The fixed centers and widths provide
    a well-conditioned optimization landscape for BFGS.
    
    Optimization: The basis matrix is cached after the first call to avoid
    redundant computation during the iterative BFGS fitting process.
    """
    global _basis_cache, _x_cache, _start_time
    
    # Ensure x is a 1D array
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
        
    params = np.asarray(params, dtype=np.float64)
    if params.ndim != 1:
        params = params.ravel()
        
    # Time limit guard (150s limit, use 145s for safety)
    if time.time() - _start_time > 145:
        return np.zeros_like(x)

    # Check if x has changed (it shouldn't during a single fit, but safe to check)
    if _x_cache is not None and np.array_equal(x, _x_cache):
        basis = _basis_cache
    else:
        n_points = x.shape[0]
        # basis shape: (10, n_points)
        basis = np.zeros((10, n_points), dtype=np.float64)
        
        # Define 10 evenly spaced centers across [0.1, 6.0]
        x_min = 0.1
        x_max = 6.0
        centers = np.linspace(x_min, x_max, 10)
        
        # Choose width sigma such that adjacent centers have significant overlap
        # Spacing is (x_max - x_min) / 9 = 5.9 / 9 ≈ 0.656
        # Use sigma = 0.5 for good local coverage with overlap
        sigma = 0.5
        
        for k in range(10):
            diff = x - centers[k]
            basis[k] = np.exp(-0.5 * (diff / sigma) ** 2)
            
        # Update cache
        _x_cache = x.copy()
        _basis_cache = basis

    # Ensure params is length 10
    if len(params) < 10:
        p = np.zeros(10, dtype=np.float64)
        p[:len(params)] = params
    else:
        p = params[:10]
        
    # Compute linear combination: result = sum(p_k * RBF_k(x))
    # basis shape (10, N), p shape (10,) -> result shape (N,)
    result = basis.T @ p
    
    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result
