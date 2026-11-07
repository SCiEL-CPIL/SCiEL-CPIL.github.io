# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a rich, fixed, linear-in-params basis.
    
    The basis combines Chebyshev polynomials (up to degree 6) and Fourier
    terms (sin/cos with frequencies 1, 2, 3) to capture curvature and
    oscillations. BFGS fits the 10 coefficients starting from all ones.
    
    Since the basis is linear in params, BFGS will find the global optimum
    of the least-squares problem efficiently.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure params has length 10
    if params.shape[0] != 10:
        # Pad or truncate
        if params.shape[0] < 10:
            params = np.pad(params, (0, 10 - params.shape[0]), constant_values=1.0)
        else:
            params = params[:10]
    
    # Build basis matrix: 10 columns
    # Chebyshev polynomials T_0 through T_5 (6 terms)
    # Fourier: sin(x), cos(x), sin(2x), cos(2x) (4 terms)
    
    # Chebyshev polynomials
    T0 = np.ones_like(x)
    T1 = x.copy()
    T2 = 2 * x**2 - 1
    T3 = 4 * x**3 - 3 * x
    T4 = 8 * x**4 - 8 * x**2 + 1
    T5 = 16 * x**5 - 20 * x**3 + 5 * x
    
    # Fourier terms
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2 * x)
    cos_2x = np.cos(2 * x)
    
    # Combine into basis matrix (n_points x 10)
    basis = np.column_stack([
        T0, T1, T2, T3, T4, T5,
        sin_x, cos_x, sin_2x, cos_2x
    ])
    
    # Linear combination
    return basis @ params
# EVOLVE-BLOCK-END
