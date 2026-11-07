"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a fixed Radial Basis Function (RBF) network.
    
    The model is:
    y = sum_{i=0}^{9} p[i] * phi_i(x)
    
    Where phi_i(x) are Gaussian RBFs centered at equally spaced points
    across the input domain [0.1, 6.0].
    
    Parameters:
    x: Input array of shape (N,)
    params: Array of 10 weights for the RBFs.
    
    This model is linear in params, ensuring convex optimization and reliable
    convergence for the evaluator's BFGS fitter. RBFs provide a flexible
    approximation basis for smooth functions, often outperforming fixed
    polynomial/trigonometric bases for arbitrary targets.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure params are finite
    if not np.all(np.isfinite(p)):
        p = np.where(np.isfinite(p), p, 0.0)
        
    # Define RBF centers and width
    # Domain is [0.1, 6.0]
    x_min = 0.1
    x_max = 6.0
    n_rbf = 10
    
    # Equally spaced centers
    centers = np.linspace(x_min, x_max, n_rbf)
    
    # Width sigma: roughly the distance between centers to ensure overlap
    # but maintain locality
    sigma = (x_max - x_min) / n_rbf
    
    # Compute RBF values
    # x is (N,), centers is (10,)
    # We want phi[i, j] = exp(-(x[i] - centers[j])^2 / (2 * sigma^2))
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    phi = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    
    # Compute prediction: y = phi @ p
    prediction = phi @ p
    
    # Check for non-finite values
    if not np.all(np.isfinite(prediction)):
        prediction = np.where(np.isfinite(prediction), prediction, 0.0)
        
    return prediction
# EVOLVE-BLOCK-END
