"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a Sparse Quadratic Polynomial on Normalized Inputs.
    
    Strategy:
    1. Normalize inputs to [0, 1] based on known training ranges to improve
       optimization conditioning. This ensures BFGS starting from ones is
       well-scaled relative to the feature magnitudes.
    2. Use a sparse quadratic polynomial basis. This captures curvature
       (parabolic effects of T and pH) and key interactions (s*T, b*T)
       without the complexity of a full mechanistic model, which may suffer
       from poor initialization of scale parameters (e.g., temperature optimum).
    
    Parameters (10 total):
    p[0]: Bias (intercept)
    p[1]: Coeff for b (density)
    p[2]: Coeff for s (substrate)
    p[3]: Coeff for temp
    p[4]: Coeff for pH
    p[5]: Coeff for temp^2
    p[6]: Coeff for pH^2
    p[7]: Coeff for s * temp
    p[8]: Coeff for b * temp
    p[9]: Coeff for s^2
    """
    # Normalization constants based on task description ranges:
    # b: [1, 4.998] -> span ~4
    # s: [1, 4.999] -> span ~4
    # temp: [1.002, 39.98] -> span ~39
    # pH: [2, 12] -> span ~10
    
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    temp_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    # Extract parameters
    w0 = params[0]
    w_b = params[1]
    w_s = params[2]
    w_t = params[3]
    w_ph = params[4]
    w_t2 = params[5]
    w_ph2 = params[6]
    w_st = params[7]
    w_bt = params[8]
    w_s2 = params[9]
    
    # Construct prediction
    prediction = (
        w0 +
        w_b * b_norm +
        w_s * s_norm +
        w_t * temp_norm +
        w_ph * ph_norm +
        w_t2 * temp_norm**2 +
        w_ph2 * ph_norm**2 +
        w_st * s_norm * temp_norm +
        w_bt * b_norm * temp_norm +
        w_s2 * s_norm**2
    )
    
    return prediction
# EVOLVE-BLOCK-END
