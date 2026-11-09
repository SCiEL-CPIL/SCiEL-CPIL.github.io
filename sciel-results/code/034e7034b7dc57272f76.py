# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model using a Chebyshev polynomial basis
    in normalized state variables (x, v).
    
    Chebyshev polynomials provide superior conditioning and global approximation
    properties for bounded nonlinear dynamics compared to monomial polynomials.
    This avoids ill-conditioning issues of high-degree monomials while maintaining
    global coverage (unlike sparse RBFs).
    
    The basis includes:
    - Pure x terms: T0, T1, T2, T3 in normalized x
    - Pure v terms: T1, T2, T3 in normalized v
    - Cross terms: T1(x)*T1(v), T1(x)*T2(v), T2(x)*T1(v)
    
    Inputs:
    t: Time
    x: Position
    v: Velocity
    params: Array of 10 fitted constants
    
    Basis Functions:
    p0: Constant offset
    p1: T1(x_n) - linear in normalized x
    p2: T1(v_n) - linear in normalized v
    p3: T2(x_n) - quadratic in normalized x
    p4: T2(v_n) - quadratic in normalized v
    p5: T3(x_n) - cubic in normalized x
    p6: T3(v_n) - cubic in normalized v
    p7: T1(x_n)*T1(v_n) - bilinear cross term
    p8: T1(x_n)*T2(v_n) - mixed cross term
    p9: T2(x_n)*T1(v_n) - mixed cross term
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize x and v to [-1, 1] for Chebyshev basis
    # x range: [-0.2056, 0.1749]
    x_min, x_max = -0.2056, 0.1749
    x_mid = (x_max + x_min) / 2.0
    x_half_range = (x_max - x_min) / 2.0
    
    # v range: [-0.3391, 0.3624]
    v_min, v_max = -0.3391, 0.3624
    v_mid = (v_max + v_min) / 2.0
    v_half_range = (v_max - v_min) / 2.0
    
    x_n = (x - x_mid) / x_half_range
    v_n = (v - v_mid) / v_half_range
    
    # Compute Chebyshev polynomials T0, T1, T2, T3
    # T0(z) = 1
    # T1(z) = z
    # T2(z) = 2z^2 - 1
    # T3(z) = 4z^3 - 3z
    
    x2 = x_n * x_n
    x3 = x2 * x_n
    v2 = v_n * v_n
    v3 = v2 * v_n
    
    T1_x = x_n
    T2_x = 2.0 * x2 - 1.0
    T3_x = 4.0 * x3 - 3.0 * x_n
    
    T1_v = v_n
    T2_v = 2.0 * v2 - 1.0
    T3_v = 4.0 * v3 - 3.0 * v_n
    
    # Construct basis functions
    f0 = np.ones_like(x)
    f1 = T1_x
    f2 = T1_v
    f3 = T2_x
    f4 = T2_v
    f5 = T3_x
    f6 = T3_v
    f7 = T1_x * T1_v
    f8 = T1_x * T2_v
    f9 = T2_x * T1_v
    
    # Linear combination of basis functions
    return (p0 * f0 + p1 * f1 + p2 * f2 + p3 * f3 + p4 * f4 +
            p5 * f5 + p6 * f6 + p7 * f7 + p8 * f8 + p9 * f9)
# EVOLVE-BLOCK-END
