"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using Chebyshev polynomials of the first kind.
    
    Maps inputs from [0.5, 2] to [-1, 1] for Chebyshev evaluation,
    then constructs a linear combination of Chebyshev basis terms.
    
    With 10 parameters, we use:
    - Constant term (T0 for all)
    - Linear terms (T1) for x, y, z
    - Quadratic cross terms (T1(x)*T1(y), T1(x)*T1(z), T1(y)*T1(z))
    - Pure quadratic terms (T2(x), T2(y), T2(z))
    
    This gives 1 + 3 + 3 + 3 = 10 parameters total.
    """
    # Map from [0.5, 2] to [-1, 1]
    # t = (2 * (val - 0.5)) / (2 - 0.5) - 1 = (2 * (val - 0.5)) / 1.5 - 1
    # Simplify: t = (4/3) * (val - 0.5) - 1 = (4/3)*val - 2/3 - 1 = (4/3)*val - 5/3
    # Or: t = 2*(val - 1.25)/0.75 - 1 = (2/0.75)*(val - 1.25) - 1 = (8/3)*(val - 1.25) - 1
    # Let's use: center = (0.5 + 2)/2 = 1.25, half_range = (2 - 0.5)/2 = 0.75
    # t = (val - center) / half_range = (val - 1.25) / 0.75
    
    center = 1.25
    half_range = 0.75
    
    tx = (x - center) / half_range
    ty = (y - center) / half_range
    tz = (z - center) / half_range
    
    # Clip to [-1, 1] to avoid numerical issues (shouldn't be needed for well-formed inputs)
    tx = np.clip(tx, -1.0, 1.0)
    ty = np.clip(ty, -1.0, 1.0)
    tz = np.clip(tz, -1.0, 1.0)
    
    # Chebyshev polynomials of the first kind
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2*t^2 - 1
    # T3(t) = 4*t^3 - 3*t
    # T4(t) = 8*t^4 - 8*t^2 + 1
    
    T0 = np.ones_like(tx)
    T1x = tx
    T1y = ty
    T1z = tz
    T2x = 2.0 * tx**2 - 1.0
    T2y = 2.0 * ty**2 - 1.0
    T2z = 2.0 * tz**2 - 1.0
    T3x = 4.0 * tx**3 - 3.0 * tx
    T3y = 4.0 * ty**3 - 3.0 * ty
    T3z = 4.0 * tz**3 - 3.0 * tz
    T4x = 8.0 * tx**4 - 8.0 * tx**2 + 1.0
    T4y = 8.0 * ty**4 - 8.0 * ty**2 + 1.0
    T4z = 8.0 * tz**4 - 8.0 * tz**2 + 1.0
    
    # Build the linear combination with 10 parameters
    # Strategy: Include more cross terms and some cubic terms for better non-linear capture
    # params[0]: constant
    # params[1]: T1(x)
    # params[2]: T1(y)
    # params[3]: T1(z)
    # params[4]: T1(x)*T1(y)
    # params[5]: T1(x)*T1(z)
    # params[6]: T1(y)*T1(z)
    # params[7]: T2(x)*T1(y)  -- mixed order cross term
    # params[8]: T2(y)*T1(z)  -- mixed order cross term  
    # params[9]: T3(x)        -- cubic pure term
    
    p = params
    
    result = (
        p[0] * T0 +
        p[1] * T1x +
        p[2] * T1y +
        p[3] * T1z +
        p[4] * (T1x * T1y) +
        p[5] * (T1x * T1z) +
        p[6] * (T1y * T1z) +
        p[7] * (T2x * T1y) +
        p[8] * (T2y * T1z) +
        p[9] * T3x
    )
    
    return result
# EVOLVE-BLOCK-END
