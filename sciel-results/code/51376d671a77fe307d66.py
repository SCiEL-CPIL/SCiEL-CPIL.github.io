"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator, approximated using a 2D Chebyshev polynomial expansion.
    
    This approach maps the input ranges to the standard Chebyshev domain [-1, 1] and uses
    orthogonal polynomial basis functions. This is distinct from trigonometric expansions
    and often provides better conditioning for smooth functions defined on bounded intervals.
    
    Basis functions (up to 3rd order in x and v, mixed terms included):
    1. 1
    2. u
    3. w
    4. T2(u)
    5. T2(w)
    6. u*w
    7. T3(u)
    8. T3(w)
    9. T2(u)*w
    10. u*T2(w)
    
    where u and w are normalized x and v to [-1, 1].
    """
    # Input ranges from the task description
    x_min, x_max = -0.5728, 0.5275
    v_min, v_max = -0.1754, 0.2821
    
    # Normalize inputs to [-1, 1]
    x_center = (x_max + x_min) / 2.0
    x_range = x_max - x_min
    v_center = (v_max + v_min) / 2.0
    v_range = v_max - v_min
    
    # Avoid division by zero
    if x_range == 0: x_range = 1e-8
    if v_range == 0: v_range = 1e-8
    
    u = (2.0 * x - x_center) / x_range
    w = (2.0 * v - v_center) / v_range
    
    # Clip to ensure numerical stability
    u = np.clip(u, -1.0, 1.0)
    w = np.clip(w, -1.0, 1.0)
    
    # Compute Chebyshev polynomials
    T0_u = np.ones_like(u)
    T0_w = np.ones_like(w)
    
    T1_u = u
    T1_w = w
    
    T2_u = 2.0 * u**2 - 1.0
    T2_w = 2.0 * w**2 - 1.0
    
    T3_u = 4.0 * u**3 - 3.0 * u
    T3_w = 4.0 * w**3 - 3.0 * w
    
    # Construct the 10 basis functions
    b0 = T0_u * T0_w       # 1
    b1 = T1_u * T0_w       # u
    b2 = T0_u * T1_w       # w
    b3 = T2_u * T0_w       # T2(u)
    b4 = T0_u * T2_w       # T2(w)
    b5 = T1_u * T1_w       # u*w
    b6 = T3_u * T0_w       # T3(u)
    b7 = T0_u * T3_w       # T3(w)
    b8 = T2_u * T1_w       # T2(u)*w
    b9 = T1_u * T2_w       # u*T2(w)
    
    # Linear combination with parameters
    p = params
    
    return p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + p[4] * b4 + \
           p[5] * b5 + p[6] * b6 + p[7] * b7 + p[8] * b8 + p[9] * b9
# EVOLVE-BLOCK-END
