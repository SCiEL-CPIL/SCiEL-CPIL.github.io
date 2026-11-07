# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator, approximated using a 2D Legendre polynomial expansion.
    
    Legendre polynomials are orthogonal with respect to the uniform distribution on [-1, 1].
    If the input data is uniformly distributed in the training ranges, this basis provides
    better conditioning and faster convergence for linear least-squares fitting compared to
    Chebyshev polynomials (which are orthogonal with respect to the arcsine distribution).
    
    Basis functions (up to 3rd order in x and v, mixed terms included):
    1. 1
    2. u
    3. w
    4. P2(u)
    5. P2(w)
    6. u*w
    7. P3(u)
    8. P3(w)
    9. P2(u)*w
    10. u*P2(w)
    
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
    
    # Clip to ensure numerical stability (mostly no-op for in-range data)
    u = np.clip(u, -1.0, 1.0)
    w = np.clip(w, -1.0, 1.0)
    
    # Compute Legendre polynomials
    # P0(z) = 1
    # P1(z) = z
    # P2(z) = (3z^2 - 1) / 2
    # P3(z) = (5z^3 - 3z) / 2
    
    P0_u = np.ones_like(u)
    P0_w = np.ones_like(w)
    
    P1_u = u
    P1_w = w
    
    P2_u = 0.5 * (3.0 * u**2 - 1.0)
    P2_w = 0.5 * (3.0 * w**2 - 1.0)
    
    P3_u = 0.5 * (5.0 * u**3 - 3.0 * u)
    P3_w = 0.5 * (5.0 * w**3 - 3.0 * w)
    
    # Construct the 10 basis functions
    b0 = P0_u * P0_w       # 1
    b1 = P1_u * P0_w       # u
    b2 = P0_u * P1_w       # w
    b3 = P2_u * P0_w       # P2(u)
    b4 = P0_u * P2_w       # P2(w)
    b5 = P1_u * P1_w       # u*w
    b6 = P3_u * P0_w       # P3(u)
    b7 = P0_u * P3_w       # P3(w)
    b8 = P2_u * P1_w       # P2(u)*w
    b9 = P1_u * P2_w       # u*P2(w)
    
    # Linear combination with parameters
    p = params
    
    return p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + p[4] * b4 + \
           p[5] * b5 + p[6] * b6 + p[7] * b7 + p[8] * b8 + p[9] * b9
# EVOLVE-BLOCK-END
