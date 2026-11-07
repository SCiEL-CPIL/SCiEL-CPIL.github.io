import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a linear combination of fixed Gaussian Radial Basis Functions.
    
    The model is:
    y(x) = sum_{i=0}^{9} c_i * exp(-((x - mu_i)^2) / (2 * sigma_i^2))
    
    The centers mu_i are fixed at symmetric points in [-2, 2] to cover the domain.
    The widths sigma_i are fixed to provide local sensitivity.
    The parameters c_i are the amplitudes, optimized by BFGS.
    
    Since the model is linear in the parameters, the optimization problem is convex,
    ensuring BFGS finds the global minimum (within numerical precision).
    """
    # Fixed centers for the RBFs, spread across the domain [-2, 2]
    # Using 10 centers: -1.8, -1.4, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.4, 1.8
    centers = np.array([-1.8, -1.4, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.4, 1.8])
    
    # Fixed widths (sigma) for the RBFs
    # A width of 0.5 provides good local coverage without being too narrow
    sigmas = np.full(10, 0.5)
    
    # Compute the basis functions: phi_j(x) = exp(-((x - mu_j)^2) / (2 * sigma_j^2))
    # x shape: (N,)
    # centers shape: (10,)
    # We need to compute (N, 10) matrix of basis values
    
    # Efficient computation:
    # diff = x[:, None] - centers[None, :]  # (N, 10)
    # basis = np.exp(-0.5 * (diff / sigmas[None, :])**2)
    
    diff = x[:, None] - centers[None, :]
    basis = np.exp(-0.5 * (diff / sigmas[None, :])**2)
    
    # Linear combination with parameters
    # params shape: (10,)
    y = basis @ params
    
    return y
