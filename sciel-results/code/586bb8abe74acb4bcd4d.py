"""Symbolic regression: equation for a nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form acceleration for a nonlinear oscillator.
    
    Uses a Chebyshev polynomial basis in normalized coordinates.
    Chebyshev polynomials are orthogonal on [-1, 1] and provide a well-conditioned
    basis for fitting, which helps BFGS converge more reliably than monomial powers.
    
    Basis terms (10 total, total degree <= 3):
    1.  T_0(x)T_0(v)
    2.  T_0(x)T_1(v)
    3.  T_0(x)T_2(v)
    4.  T_0(x)T_3(v)
    5.  T_1(x)T_0(v)
    6.  T_1(x)T_1(v)
    7.  T_1(x)T_2(v)
    8.  T_2(x)T_0(v)
    9.  T_2(x)T_1(v)
    10. T_3(x)T_0(v)
    
    where T_n is the Chebyshev polynomial of the first kind.
    Inputs are normalized to roughly [-1, 1].
    """
    # Normalize inputs to O(1) scale based on training range max abs values
    # x range: [-0.5728, 0.5275] -> max abs ~ 0.57
    # v range: [-0.1754, 0.2821] -> max abs ~ 0.28
    # Using slightly wider scaling to ensure values stay within [-1, 1] mostly
    x_s = x / 0.60
    v_s = v / 0.30
    
    # Monomial basis, excluding x^2 to avoid potential collinearity/singularity
    # and include x^4 to capture stronger nonlinearity if needed.
    # Terms:
    # 1. 1
    # 2. x
    # 3. v
    # 4. x*v
    # 5. v^2
    # 6. x^3
    # 7. x^2*v
    # 8. x*v^2
    # 9. v^3
    # 10. x^4
    
    b0 = np.ones_like(x_s)
    b1 = x_s
    b2 = v_s
    b3 = x_s * v_s
    b4 = v_s * v_s
    b5 = x_s * x_s * x_s
    b6 = x_s * x_s * v_s
    b7 = x_s * v_s * v_s
    b8 = v_s * v_s * v_s
    b9 = x_s * x_s * x_s * x_s
    
    # Linear combination of basis functions
    return (params[0] * b0 + params[1] * b1 + params[2] * b2 +
            params[3] * b3 + params[4] * b4 + params[5] * b5 +
            params[6] * b6 + params[7] * b7 + params[8] * b8 +
            params[9] * b9)
# EVOLVE-BLOCK-END
