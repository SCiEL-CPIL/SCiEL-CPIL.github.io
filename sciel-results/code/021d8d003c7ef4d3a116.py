"""Symbolic regression using Gaussian Radial Basis Functions."""
# EVOLVE-BLOCK-START
import numpy as np

def _rbf_basis(x: np.ndarray) -> np.ndarray:
    """
    Build a basis matrix of Gaussian Radial Basis Functions.
    Centers are fixed at 10 points in [-1.5, 1.5].
    Width (sigma) is fixed to 0.5.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    # Fixed centers
    centers = np.linspace(-1.5, 1.5, 10)
    # Fixed width
    sigma = 0.5
    
    # Compute distances
    # x shape: (n, 1)
    # centers shape: (10, 1)
    # diff shape: (n, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian: exp(-0.5 * (diff/sigma)^2)
    basis = np.exp(-0.5 * (diff / sigma) ** 2)
    
    return basis

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: y = sum_{i=0}^{9} p_i * G(x - c_i)
    where G is a Gaussian RBF.
    
    This model is linear in parameters. BFGS will find the least-squares solution.
    The fixed centers and width provide a stable, well-conditioned basis for smooth functions.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if len(params) != 10:
        # Fallback if params length is wrong (should not happen)
        return np.zeros_like(x)
        
    basis = _rbf_basis(x)
    
    # Linear combination
    result = basis @ params
    
    # Ensure finite output
    return np.nan_to_num(result, nan=0.0, posinf=1e8, neginf=-1e8)
# EVOLVE-BLOCK-END
