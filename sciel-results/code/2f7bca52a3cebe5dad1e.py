import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Radial Basis Function basis.
    10 Gaussian kernels centered at distinct points in [-2, 2].
    Centers: -1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8 (7 centers)
    Plus 3 additional centers to fill 10: -0.3, 0.3, 1.5
    Actually, let's use 10 evenly spaced centers from -1.8 to 1.8.
    Width (sigma) is fixed to provide good coverage.
    """
    # 10 centers evenly spaced in [-1.8, 1.8]
    centers = np.array([-1.8, -1.44, -1.08, -0.72, -0.36, 0.0, 0.36, 0.72, 1.08, 1.44])
    
    # Fixed width for Gaussian kernels
    sigma = 0.8
    
    # Compute Gaussian RBFs: exp(-0.5 * ((x - center) / sigma)^2)
    # Vectorized computation
    # x has shape (N,), centers has shape (10,)
    # We want result shape (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    basis = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination of basis functions with params
    return basis @ params
