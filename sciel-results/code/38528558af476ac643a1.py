"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map input domain [-3, 3] to standard Chebyshev interval [-1, 1]
    t = x / 3.0
    
    # Precompute Chebyshev polynomials of degree 0 through 9
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
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
    
    # Normalize each basis function to have unit RMS on the current input x.
    # This improves the conditioning of the design matrix for BFGS,
    # allowing for faster and more accurate convergence to the optimal parameters.
    def normalize_basis(basis):
        rms = np.sqrt(np.mean(basis**2))
        if rms < 1e-12:
            return basis
        return basis / rms
        
    B0 = normalize_basis(T0)
    B1 = normalize_basis(T1)
    B2 = normalize_basis(T2)
    B3 = normalize_basis(T3)
    B4 = normalize_basis(T4)
    B5 = normalize_basis(T5)
    B6 = normalize_basis(T6)
    B7 = normalize_basis(T7)
    B8 = normalize_basis(T8)
    B9 = normalize_basis(T9)
    
    # Linear combination with 10 parameters
    result = (params[0] * B0 +
              params[1] * B1 +
              params[2] * B2 +
              params[3] * B3 +
              params[4] * B4 +
              params[5] * B5 +
              params[6] * B6 +
              params[7] * B7 +
              params[8] * B8 +
              params[9] * B9)
    
    return result
# EVOLVE-BLOCK-END
