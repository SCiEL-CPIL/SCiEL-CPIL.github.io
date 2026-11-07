"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Softplus basis expansion with fixed centers and slopes.
    params: array of 10 weights.
    Basis: sum_{i=0}^{9} w_i * softplus(s_i * (x - c_i))
    Centers c_i are evenly spaced in [-2, 2].
    Slopes s_i are fixed to a reasonable value (e.g., 2.0) to balance smoothness and flexibility.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # 10 evenly spaced centers in [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Fixed slopes: use a moderate slope to capture non-linear features
    # A slope of 2.0 gives a reasonable transition width
    slope = 2.0
    
    # Compute softplus basis: softplus(z) = log(1 + exp(z))
    # For numerical stability, use np.logaddexp(0, z)
    z = slope * (x[:, np.newaxis] - centers[np.newaxis, :])
    basis = np.logaddexp(0.0, z)  # shape (n_points, 10)
    
    # Linear combination with weights
    return basis @ params
# EVOLVE-BLOCK-END
