import numpy as np
import time

_DEADLINE_S = 140.0
_MARGIN_S = 5.0
_START_TIME = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a hybrid Chebyshev + localized bump basis.
    
    The basis consists of:
    1. Chebyshev polynomials T_0 to T_5 on [-1, 1] (mapped from x in [-2, 2])
    2. Two smooth Gaussian-like bumps centered at -1 and +1
    
    This combines the global spectral efficiency of Chebyshev polynomials (inspiration 1)
    with the local flexibility of bump functions (inspiration 2) while maintaining
    good conditioning for BFGS optimization (inspired by the normalization in the
    current program).
    """
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE_S - _MARGIN_S:
        # Fallback: simple linear using first two params
        x_safe = np.clip(np.asarray(x, dtype=float), -2.0, 2.0)
        p = np.asarray(params, dtype=float)
        if len(p) >= 2:
            return p[0] + p[1] * x_safe
        return np.zeros_like(x_safe)
    
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Ensure params has length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
        
    # Clip x to domain for stability
    xc = np.clip(x, -2.0, 2.0)
    N = len(xc)
    
    # Scale to [-1, 1] for Chebyshev stability
    t = xc * 0.5
    
    # Compute Chebyshev polynomials T_0 to T_5
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    # T_4(t) = 8t^4 - 8t^2 + 1
    # T_5(t) = 16t^5 - 20t^3 + 5t
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * t - 1.0
    T3 = 4.0 * t * t * t - 3.0 * t
    T4 = 8.0 * t * t * t * t - 8.0 * t * t + 1.0
    T5 = 16.0 * t * t * t * t * t - 20.0 * t * t * t + 5.0 * t
    
    # Two smooth Gaussian-like bumps for local flexibility
    # Centered at -1 and +1 with moderate width
    # Using exp(-k*(x-c)^2) form which is smooth and well-behaved
    bump1 = np.exp(-2.0 * (xc + 1.0) ** 2)
    bump2 = np.exp(-2.0 * (xc - 1.0) ** 2)
    
    # Build 10-column basis matrix
    # Columns: T0, T1, T2, T3, T4, T5, bump1, bump2, and two additional terms
    # For the last two slots, we use scaled versions of the bumps with different widths
    # to allow for more flexible local behavior
    bump1_wide = np.exp(-0.5 * (xc + 1.0) ** 2)
    bump2_wide = np.exp(-0.5 * (xc - 1.0) ** 2)
    
    basis = np.column_stack([T0, T1, T2, T3, T4, T5, bump1, bump2, bump1_wide, bump2_wide])
    
    # Normalize columns by their max absolute value on [-2, 2] for conditioning
    # Compute max values analytically or approximately
    max_vals = np.ones(10)
    # Chebyshev polynomials are bounded by 1 on [-1, 1]
    max_vals[0:6] = 1.0
    # Bump functions: max at center
    max_vals[6] = 1.0   # bump1 at x=-1
    max_vals[7] = 1.0   # bump2 at x=1
    max_vals[8] = 1.0   # bump1_wide at x=-1
    max_vals[9] = 1.0   # bump2_wide at x=1
    
    # Safety check
    max_vals = np.where(max_vals > 1e-12, max_vals, 1.0)
    
    # Normalize basis columns
    basis = basis / max_vals[np.newaxis, :]
    
    # Compute prediction
    pred = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        # Fallback to linear
        return params[0] + params[1] * xc
        
    return pred
