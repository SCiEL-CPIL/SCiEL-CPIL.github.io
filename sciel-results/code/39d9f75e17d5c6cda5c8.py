import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using Gaussian RBF basis with fixed centers.
    
    10 RBF centers are placed across the 2D (x, v) training domain.
    Each parameter is the amplitude of one RBF. This provides a flexible
    nonlinear model that BFGS can fit the 10 amplitudes for.
    """
    # Fixed RBF centers spread across training domain
    # x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
    centers_x = np.array([
        -0.45, 0.0, 0.45,
        -0.25, 0.25,
        0.0, -0.4, 0.4,
        -0.5, 0.5
    ])
    centers_v = np.array([
        -0.10, 0.0, 0.15,
        0.0, 0.0,
        -0.15, 0.10, -0.05,
        0.05, -0.10
    ])
    
    # Fixed width for RBFs (sigma)
    sigma = 0.3
    
    # Compute RBF values: shape (n_samples, 10)
    dx = x[:, np.newaxis] - centers_x[np.newaxis, :]
    dv = v[:, np.newaxis] - centers_v[np.newaxis, :]
    
    r2 = dx**2 + dv**2
    rbf = np.exp(-r2 / (2 * sigma**2))
    
    # Linear combination with fitted amplitudes
    return rbf @ params
