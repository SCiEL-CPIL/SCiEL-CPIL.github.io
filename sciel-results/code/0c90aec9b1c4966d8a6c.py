import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Use 10 localized Gaussian RBFs with fixed centers and widths
    # Centers spread across [-3, 3], widths varied to capture different scales
    
    if len(x) == 0:
        return np.array([])
        
    # Define centers and widths for 10 RBFs
    # Centers: evenly spaced from -2.7 to 2.7 (avoiding boundaries slightly)
    centers = np.linspace(-2.7, 2.7, 10)
    # Widths: vary to allow both narrow and broad features
    # Use a mix: some narrow (0.3), some medium (0.6), some broad (1.2)
    widths = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0])
    
    # Compute RBF basis: phi_j(x) = exp(-((x - c_j)^2) / (2 * sigma_j^2))
    # x shape: (N,), centers shape: (10,)
    # Compute squared distances: (N, 1) - (1, 10) -> (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    sq_diff = diff * diff
    
    # sigma^2 = widths^2
    sigma_sq = widths[np.newaxis, :] ** 2
    
    # phi = exp(-sq_diff / (2 * sigma_sq))
    basis = np.exp(-sq_diff / (2.0 * sigma_sq))
    
    return basis @ params
