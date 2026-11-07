"""Program using RBF kernel basis for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form symbolic regression candidate using Gaussian RBF kernels.
    10 RBF kernels with fixed, evenly spaced centers over the input domain [-3, 3].
    The 10 parameters act as adaptive weights for local interpolation.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Fixed centers evenly spaced over [-3, 3]
    # 10 centers from -3 to 3
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Bandwidth: use a value that provides good local coverage
    # With 10 centers over range 6, spacing is 6/9 ≈ 0.67
    # Use sigma such that adjacent kernels have moderate overlap
    sigma = 0.8
    
    # Compute RBF values: phi_i(x) = exp(-(x - c_i)^2 / (2*sigma^2))
    # Shape: (n_points, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination: y = sum(params[i] * rbf_values[:, i])
    y = rbf_values @ params
    
    # Ensure finite output
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
# EVOLVE-BLOCK-END
