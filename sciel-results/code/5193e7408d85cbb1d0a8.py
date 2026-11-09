"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map input domain [-3, 3] to standard Chebyshev interval [-1, 1]
    t = x / 3.0
    
    # Precompute Chebyshev polynomials of degree 0 through 9
    # Using stable recurrence
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    T5 = 2.0 * t * T4 - T3
    T6 = 2.0 * t * T5 - T4
    T7 = 2.0 * t * T6 - T5
    T8 = 2.0 * t * T7 - T6
    T9 = 2.0 * t * T8 - T7
    
    # Normalize each basis function to have similar scale for better conditioning
    # This helps BFGS converge more reliably
    norm_T0 = np.sqrt(np.mean(T0**2))
    norm_T1 = np.sqrt(np.mean(T1**2))
    norm_T2 = np.sqrt(np.mean(T2**2))
    norm_T3 = np.sqrt(np.mean(T3**2))
    norm_T4 = np.sqrt(np.mean(T4**2))
    norm_T5 = np.sqrt(np.mean(T5**2))
    norm_T6 = np.sqrt(np.mean(T6**2))
    norm_T7 = np.sqrt(np.mean(T7**2))
    norm_T8 = np.sqrt(np.mean(T8**2))
    norm_T9 = np.sqrt(np.mean(T9**2))
    
    # Avoid division by zero
    eps = 1e-12
    norm_T0 = max(norm_T0, eps)
    norm_T1 = max(norm_T1, eps)
    norm_T2 = max(norm_T2, eps)
    norm_T3 = max(norm_T3, eps)
    norm_T4 = max(norm_T4, eps)
    norm_T5 = max(norm_T5, eps)
    norm_T6 = max(norm_T6, eps)
    norm_T7 = max(norm_T7, eps)
    norm_T8 = max(norm_T8, eps)
    norm_T9 = max(norm_T9, eps)
    
    # Normalize basis functions
    T0 = T0 / norm_T0
    T1 = T1 / norm_T1
    T2 = T2 / norm_T2
    T3 = T3 / norm_T3
    T4 = T4 / norm_T4
    T5 = T5 / norm_T5
    T6 = T6 / norm_T6
    T7 = T7 / norm_T7
    T8 = T8 / norm_T8
    T9 = T9 / norm_T9
    
    # Linear combination with 10 parameters
    result = (params[0] * T0 +
              params[1] * T1 +
              params[2] * T2 +
              params[3] * T3 +
              params[4] * T4 +
              params[5] * T5 +
              params[6] * T6 +
              params[7] * T7 +
              params[8] * T8 +
              params[9] * T9)
    
    return result
# EVOLVE-BLOCK-END
