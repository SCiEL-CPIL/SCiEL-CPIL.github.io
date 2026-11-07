"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate using localized Gaussian Radial Basis Functions (RBFs).
    
    Strategy:
    Instead of using global basis functions like polynomials or trigonometric functions,
    which can suffer from ill-conditioning and poor approximation properties for arbitrary
    smooth functions, we use a set of 10 Gaussian kernels.
    
    Each kernel is centered at a specific point in the domain [-3, 3].
    The width of each Gaussian is fixed to allow sufficient overlap for smooth approximation
    but distinct enough to localize features.
    
    The 10 parameters correspond to the weights (amplitudes) of these 10 Gaussian kernels.
    Since the centers and widths are fixed, the function is a linear combination of fixed
    basis functions. This makes the optimization problem convex with respect to the parameters
    (if we were using least squares directly), but even with BFGS minimizing MSE, the landscape
    is much smoother and less prone to local minima compared to non-linear basis functions.
    
    Centers are chosen to be evenly spaced across the domain [-3, 3]:
    c_i = -3 + i * (6 / 9) for i in 0..9
    This gives centers at: -3, -2.33, -1.67, -1, -0.33, 0.33, 1, 1.67, 2.33, 3
    
    Width (sigma) is chosen to be roughly half the spacing to ensure good coverage.
    Spacing = 0.666. Sigma = 0.5 seems reasonable.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Ensure p has length 10
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant')
    else:
        p = p[:10]
        
    # Define fixed centers
    # 10 centers evenly spaced from -3 to 3
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Define fixed width (sigma)
    # A sigma of 0.5 provides a balance between local flexibility and smoothness.
    sigma = 0.5
    sigma_sq = sigma * sigma
    
    # Compute the RBF values
    # x shape: (N,)
    # centers shape: (10,)
    # We want a matrix of shape (N, 10) where RBF[i, j] = exp(-0.5 * (x[i] - c[j])^2 / sigma^2)
    
    # Use broadcasting: x[:, None] - centers[None, :]
    diff = x[:, None] - centers[None, :]
    
    # Calculate squared differences
    diff_sq = diff * diff
    
    # Calculate Gaussian values
    # exp(-diff_sq / (2 * sigma^2))
    rbf_values = np.exp(-diff_sq / (2.0 * sigma_sq))
    
    # The prediction is the dot product of the RBF matrix (N, 10) and the params (10,)
    pred = rbf_values @ p
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
