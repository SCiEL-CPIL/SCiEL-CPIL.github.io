"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: low-degree polynomial + tunable sigmoid units.
    y(x) = a0 + a1*x + a2*x^2 + sum_{i=1}^{3} c_i * sigmoid(w_i * x + b_i)
    
    This combines global polynomial behavior with local nonlinear features.
    The sigmoid units can approximate peaks, steps, and smooth transitions.
    """
    # Unpack parameters
    a0, a1, a2, w1, b1, c1, w2, b2, c2, w3, b3, c3 = params[:12] if len(params) >= 12 else [
        params[0], params[1], params[2], params[3], params[4], params[5],
        params[6], params[7], params[8], params[9], 0.0, 0.0
    ]
    
    # Polynomial part
    poly = a0 + x * (a1 + x * a2)
    
    # Sigmoid part: use stable sigmoid computation
    def stable_sigmoid(z):
        # For large positive z, sigmoid(z) ≈ 1
        # For large negative z, sigmoid(z) ≈ 0
        # Use np.where to avoid overflow in exp
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
    
    sig1 = stable_sigmoid(w1 * x + b1)
    sig2 = stable_sigmoid(w2 * x + b2)
    sig3 = stable_sigmoid(w3 * x + b3)
    
    result = poly + c1 * sig1 + c2 * sig2 + c3 * sig3
    
    return result
# EVOLVE-BLOCK-END
