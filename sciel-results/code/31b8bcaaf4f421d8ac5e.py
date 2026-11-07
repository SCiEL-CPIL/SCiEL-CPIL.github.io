"""RBF-based symbolic regression for nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator using a Gaussian RBF basis.
    
    Inputs:
        t: time, range [30, 50]
        x: position, range [-0.2056, 0.1749]
        v: velocity, range [-0.3391, 0.3624]
        params: array of 10 fitted constants
    
    Architecture:
        - 3 Gaussian RBF centers in (t, x, v) space with fixed anisotropic widths
        - 3 RBF weights (params[0:3])
        - 3 width scales (params[3:6]) - one per center, applied isotropically in normalized space
        - 2 linear terms: x and v (params[6:8])
        - 1 quadratic term: x*v (params[8])
        - 1 bias (params[9])
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    # Normalize inputs to [0, 1] range for better conditioning
    t_min, t_max = 30.0, 50.0
    x_min, x_max = -0.2056, 0.1749
    v_min, v_max = -0.3391, 0.3624
    
    t_norm = (t - t_min) / (t_max - t_min)
    x_norm = (x - x_min) / (x_max - x_min)
    v_norm = (v - v_min) / (v_max - v_min)
    
    # Define RBF centers in normalized space (distinct, spread across the domain)
    # Center 1: lower region
    tc1, xc1, vc1 = 0.25, 0.35, 0.30
    # Center 2: middle region
    tc2, xc2, vc2 = 0.50, 0.50, 0.50
    # Center 3: upper region
    tc3, xc3, vc3 = 0.75, 0.65, 0.70
    
    # Width parameters: params[3:6], one per center
    # Use softplus to ensure positive widths
    w1 = np.exp(params[3])  # width for center 1
    w2 = np.exp(params[4])  # width for center 2
    w3 = np.exp(params[5])  # width for center 3
    
    # Compute squared distances in normalized space
    d1_sq = (t_norm - tc1) ** 2 + (x_norm - xc1) ** 2 + (v_norm - vc1) ** 2
    d2_sq = (t_norm - tc2) ** 2 + (x_norm - xc2) ** 2 + (v_norm - vc2) ** 2
    d3_sq = (t_norm - tc3) ** 2 + (x_norm - xc3) ** 2 + (v_norm - vc3) ** 2
    
    # Gaussian RBFs with learned widths
    rbf1 = np.exp(-d1_sq / (2.0 * w1 ** 2))
    rbf2 = np.exp(-d2_sq / (2.0 * w2 ** 2))
    rbf3 = np.exp(-d3_sq / (2.0 * w3 ** 2))
    
    # Linear and quadratic terms in original space for physical interpretability
    lin_x = x
    lin_v = v
    quad_xv = x * v
    
    # Final prediction
    pred = (params[0] * rbf1 +
            params[1] * rbf2 +
            params[2] * rbf3 +
            params[6] * lin_x +
            params[7] * lin_v +
            params[8] * quad_xv +
            params[9])
    
    return pred
# EVOLVE-BLOCK-END
