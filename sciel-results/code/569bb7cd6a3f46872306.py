"""Symbolic regression with 1 input."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a hybrid basis of 10 terms.
    
    Combines low-order Chebyshev polynomials (for global smooth trends) with
    Radial Basis Functions (RBFs) with fixed centers (for local variations).
    This hybrid approach captures both global structure and local features,
    improving fit quality for diverse target functions compared to pure
    polynomial bases.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure x is 1D for consistent processing
    x_flat = x.flatten()
    n_points = x_flat.size
    
    # 1. Chebyshev Basis (T_0 to T_4)
    # Map x from [-3, 3] to t in [-1, 1]
    t = x_flat / 3.0
    t = np.clip(t, -1.0, 1.0) # Clip for safety in Chebyshev recurrence
    
    cheb_basis = np.empty((5, n_points), dtype=np.float64)
    cheb_basis[0] = np.ones(n_points)
    cheb_basis[1] = t.copy()
    
    for k in range(2, 5):
        cheb_basis[k] = 2.0 * t * cheb_basis[k-1] - cheb_basis[k-2]
    
    # 2. RBF Basis (5 centers)
    # Fixed centers spread across the domain [-3, 3]
    centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    sigma = 1.0 # Width parameter
    
    rbf_basis = np.empty((5, n_points), dtype=np.float64)
    for i, c in enumerate(centers):
        # Gaussian RBF: exp(-((x - c)^2) / (2 * sigma^2))
        rbf_basis[i] = np.exp(-((x_flat - c) ** 2) / (2.0 * sigma ** 2))
    
    # Combine basis matrices
    # Total 10 basis functions
    basis = np.vstack((cheb_basis, rbf_basis))
    
    # Build the linear combination: result = basis.T @ params
    result = basis.T @ params
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result.reshape(x.shape)
# EVOLVE-BLOCK-END
