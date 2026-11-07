"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a hybrid Chebyshev + RBF basis.
    
    The model combines:
    - 4 Chebyshev polynomials T_0..T_3 (global low-frequency trends)
    - 6 Gaussian Radial Basis Functions (local high-frequency features)
    
    This allows the 10 parameters to capture both global structure and
    local variations that a pure polynomial or pure RBF might miss.
    
    RBF centers are placed at distinct points in [-3, 3]:
    -3.0, -1.5, 0.0, 1.5, 3.0, and one additional at 0.0 with different width.
    Actually, let's use 6 centers: -3.0, -1.8, -0.6, 0.6, 1.8, 3.0
    with varying widths to allow local flexibility.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Map x from [-3, 3] to t in [-1, 1] for Chebyshev
    t = x / 3.0
    
    # --- Chebyshev part (4 basis functions) ---
    # T_0 = 1, T_1 = t, T_2 = 2t^2 - 1, T_3 = 4t^3 - 3t
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * t - 1.0
    T3 = 4.0 * t * t * t - 3.0 * t
    
    cheb_part = params[0] * T0 + params[1] * T1 + params[2] * T2 + params[3] * T3
    
    # --- RBF part (6 basis functions) ---
    # Centers in [-3, 3]
    centers = np.array([-3.0, -1.8, -0.6, 0.6, 1.8, 3.0])
    # Widths: mix of narrow and medium to capture different local scales
    widths = np.array([0.8, 0.6, 0.5, 0.5, 0.6, 0.8])
    
    rbf_part = np.zeros_like(x)
    for i in range(6):
        # Gaussian RBF: exp(-((x - c_i)^2) / (2 * sigma_i^2))
        diff = x - centers[i]
        rbf_i = np.exp(-(diff * diff) / (2.0 * widths[i] * widths[i]))
        rbf_part += params[4 + i] * rbf_i
    
    result = cheb_part + rbf_part
    return result
# EVOLVE-BLOCK-END
