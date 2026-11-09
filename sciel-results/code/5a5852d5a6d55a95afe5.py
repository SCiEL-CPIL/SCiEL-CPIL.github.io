# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = None
_RBF_CACHE = {}

def _get_rbf_basis(x, y):
    """
    Computes the RBF kernel matrix for fixed centers.
    Centers are a 4x3 grid in [-1, 1] x [-1, 1].
    Returns a matrix of shape (N, 10).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Cache key: since x and y are fixed during the optimization loop, 
    # we can cache based on their memory addresses and shapes.
    # If the evaluator creates new arrays each time, this will miss, 
    # but the computation is O(N*10) which is very fast anyway.
    key = (id(x), id(y), x.shape, y.shape)
    
    if key in _RBF_CACHE:
        return _RBF_CACHE[key]
        
    n = x.shape[0]
    if n == 0:
        return np.ones((0, 10))
        
    # Define 10 fixed centers in a 4x3 grid
    # This covers the domain [-1, 1] x [-1, 1] reasonably well
    x_centers = np.array([-1.0, -0.333333, 0.333333, 1.0])
    y_centers = np.array([-1.0, 0.0, 1.0])
    
    cx, cy = np.meshgrid(x_centers, y_centers, indexing='ij')
    cx_flat = cx.flatten() # length 12
    cy_flat = cy.flatten()
    
    # We need exactly 10 centers for the 10 params.
    # Let's pick a specific subset or just use the first 10? 
    # Better: Use a 5x2 grid or similar to get exactly 10?
    # 5x2 grid:
    x_centers_5 = np.linspace(-1, 1, 5)
    y_centers_2 = np.linspace(-1, 1, 2)
    cx, cy = np.meshgrid(x_centers_5, y_centers_2, indexing='ij')
    cx_flat = cx.flatten()
    cy_flat = cy.flatten()
    
    # Compute squared Euclidean distances: ||(x,y) - (cx,cy)||^2
    # x shape (N,), cx_flat shape (10,)
    # dist_sq[i, j] = (x[i] - cx_flat[j])^2 + (y[i] - cy_flat[j])^2
    dx = x[:, np.newaxis] - cx_flat[np.newaxis, :]
    dy = y[:, np.newaxis] - cy_flat[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    # Gaussian kernel: exp(-alpha * dist_sq)
    # Choose alpha to control the width. 
    # Domain size is ~2. Distance max is ~2.82.
    # If alpha=1, exp(-1) ~ 0.37, exp(-8) ~ 0.0003.
    # This provides good localization without being too narrow.
    alpha = 1.0 
    basis = np.exp(-alpha * dist_sq)
    
    # Ensure finite values
    basis = np.nan_to_num(basis, nan=0.0, posinf=0.0, neginf=0.0)
    
    _RBF_CACHE[key] = basis
    return basis

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    # Deadline guard
    # The evaluator kills at 150s. BFGS should be fast, but let's be safe.
    # If we exceed 140s, we might want to force a return, but we can't easily 
    # interrupt the optimizer from here. We just ensure the computation is minimal.
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Ensure params is length 10
    if params.shape[0] != 10:
        if params.shape[0] < 10:
            params = np.pad(params, (0, 10 - params.shape[0]))
        else:
            params = params[:10]
            
    # Get the RBF basis
    basis = _get_rbf_basis(x, y)
    
    # Prediction is a linear combination of the RBF kernels
    pred = basis @ params
    
    # Ensure finite
    if not np.all(np.isfinite(pred)):
        # Fallback to mean of y if something goes wrong
        return np.full_like(y, np.mean(y), dtype=float)
        
    return pred
# EVOLVE-BLOCK-END
