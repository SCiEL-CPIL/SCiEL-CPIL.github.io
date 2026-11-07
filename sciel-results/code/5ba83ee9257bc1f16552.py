"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Domain is [0.1, 6]. Use 10 fixed centers evenly spaced across the domain.
    # params: length 10, interpreted as log-weights for each kernel center.
    # We use a sum of squared-exponential (RBF) kernels with a fixed bandwidth.
    # To ensure positivity and numerical stability, we use exp(params) as weights.
    # The bandwidth is fixed to a reasonable value for the domain scale.
    
    # Fixed centers evenly spaced in [0.1, 6]
    centers = np.linspace(0.1, 6.0, 10)
    
    # Fixed bandwidth: choose sigma such that the kernel has reasonable support.
    # Domain width is ~5.9. With 10 centers, spacing is ~0.65.
    # Let's use sigma = 0.8 for smooth interpolation.
    sigma = 0.8
    
    # x shape: (n,)
    # centers shape: (10,)
    # Compute distances: (n, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Squared exponential kernel: exp(-diff^2 / (2 * sigma^2))
    kernel = np.exp(-diff ** 2 / (2.0 * sigma ** 2))
    
    # Weights from params: use exp(params) to ensure positive weights
    # This allows the optimizer to adjust the magnitude of each kernel's contribution.
    weights = np.exp(params)
    
    # Prediction: sum of weighted kernels
    # (n, 10) @ (10,) -> (n,)
    pred = kernel @ weights
    
    return pred
# EVOLVE-BLOCK-END
