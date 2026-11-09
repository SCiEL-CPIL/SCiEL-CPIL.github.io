"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT_MARGIN = 148.0  # seconds, keep a safety margin before 150s kill


def _time_left():
    return _start_time + _TIMEOUT_MARGIN - time.time()


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters that tries to recover an unknown function.
    Strategy: Bivariate Chebyshev Polynomial Basis.
    
    We use a tensor product of Chebyshev polynomials T_n(x) and T_m(y)
    for n, m in {0, 1, 2}. This provides 9 basis functions:
    T0(x)T0(y), T1(x)T0(y), T2(x)T0(y), T0(x)T1(y), T1(x)T1(y), 
    T2(x)T1(y), T0(x)T2(y), T1(x)T2(y), T2(x)T2(y).
    
    The 10th parameter is a global offset or can be used to scale the 
    entire function if the target has a large DC component not captured 
    well by the mean-centered Chebyshev basis, or simply as a redundant 
    parameter that BFGS can zero out. Actually, T0(x)T0(y) is constant 1, 
    so it acts as the offset. 
    Wait, T0(x) = 1. So the first basis function is 1. 
    This means the model is:
    f(x,y) = c00*T0*T0 + c10*T1*T0 + c20*T2*T0 + c01*T0*T1 + c11*T1*T1 + 
             c21*T2*T1 + c02*T0*T2 + c12*T1*T2 + c22*T2*T2
             
    This is a polynomial of total degree 4 (since T2 is degree 2, T2*T2 is degree 4).
    It is smooth and globally defined.
    Chebyshev polynomials are orthogonal on [-1, 1], which helps with conditioning.
    
    The 10th parameter (p[9]) is currently unused in the strict 3x3 grid.
    To utilize all 10 parameters meaningfully and potentially improve flexibility,
    we can add a 10th term. A common choice is to include a higher-order term 
    or a different basis. However, mixing orders might hurt conditioning.
    
    Let's stick to the 3x3 Chebyshev grid (9 params) and use the 10th param 
    as a multiplier for the identity or a specific high-frequency term?
    Actually, if we just use 9 params, the 10th is ignored. 
    It is better to use a basis that naturally has 10 terms or use the 10th 
    param to scale one of the terms or add a small regularization-like term?
    
    Let's try a different configuration:
    Use Chebyshev T0, T1, T2 for x and T0, T1 for y? That's 6 terms.
    Use T0, T1, T2, T3 for x and T0 for y? That's 4 terms.
    
    The 3x3 grid (T0..T2 x T0..T2) is the most symmetric and complete 
    low-order representation. 
    If we must use 10 parameters, we can add T3(x)*T0(y) as the 10th term?
    Or T0(x)*T3(y)?
    Or T3(x)*T1(y)?
    
    Let's use the 3x3 grid (9 params) and the 10th param as a coefficient 
    for a "correction" term, e.g., T3(x)*T0(y) which captures higher 
    frequency behavior in x.
    
    Basis:
    0: T0(x)T0(y) = 1
    1: T1(x)T0(y) = x
    2: T2(x)T0(y) = 2x^2 - 1
    3: T0(x)T1(y) = y
    4: T1(x)T1(y) = xy
    5: T2(x)T1(y) = (2x^2 - 1)y
    6: T0(x)T2(y) = 2y^2 - 1
    7: T1(x)T2(y) = x(2y^2 - 1)
    8: T2(x)T2(y) = (2x^2 - 1)(2y^2 - 1)
    9: T3(x)T0(y) = 4x^3 - 3x  (Added for 10th param)
    
    This set is linearly independent.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # Safety check for time
    if _time_left() < 0.1:
        return np.zeros_like(x)

    # Compute Chebyshev polynomials T0, T1, T2, T3
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2t^2 - 1
    # T3(t) = 4t^3 - 3t
    
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x * x - 1.0
    T3_x = 4.0 * x * x * x - 3.0 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y * y - 1.0
    T3_y = 4.0 * y * y * y - 3.0 * y
    
    # Construct basis vectors
    # p[0] * T0x*T0y
    # p[1] * T1x*T0y
    # p[2] * T2x*T0y
    # p[3] * T0x*T1y
    # p[4] * T1x*T1y
    # p[5] * T2x*T1y
    # p[6] * T0x*T2y
    # p[7] * T1x*T2y
    # p[8] * T2x*T2y
    # p[9] * T3x*T0y
    
    pred = (
        params[0] * T0_x * T0_y +
        params[1] * T1_x * T0_y +
        params[2] * T2_x * T0_y +
        params[3] * T0_x * T1_y +
        params[4] * T1_x * T1_y +
        params[5] * T2_x * T1_y +
        params[6] * T0_x * T2_y +
        params[7] * T1_x * T2_y +
        params[8] * T2_x * T2_y +
        params[9] * T3_x * T0_y
    )
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(x)
        
    return pred
# EVOLVE-BLOCK-END
