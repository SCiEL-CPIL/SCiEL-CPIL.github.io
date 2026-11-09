# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model using Chebyshev polynomial basis
    in normalized state variables (x, v) plus sinusoidal time forcing.
    
    Chebyshev polynomials provide superior conditioning compared to monomials.
    The basis includes:
    - Chebyshev terms in normalized x (T0, T1, T2, T3)
    - Chebyshev terms in normalized v (T1, T2)
    - Cross term T1_x * T1_v for x-v coupling
    - Sinusoidal and cosinusoidal forcing in centered time
    
    Inputs:
    t: Time [30, 50]
    x: Position [-0.2056, 0.1749]
    v: Velocity [-0.3391, 0.3624]
    params: Array of 10 fitted constants
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize x and v to [-1, 1] for Chebyshev evaluation
    # x range: [-0.2056, 0.1749], mid = -0.01535, half_range = 0.19025
    x_mid = -0.01535
    x_half = 0.19025
    x_norm = (x - x_mid) / x_half
    
    # v range: [-0.3391, 0.3624], mid = 0.01165, half_range = 0.35075
    v_mid = 0.01165
    v_half = 0.35075
    v_norm = (v - v_mid) / v_half
    
    # Center time: [30, 50], mid = 40
    t_c = t - 40.0
    
    # Chebyshev polynomials T_n(z)
    # T0 = 1
    # T1 = z
    # T2 = 2z^2 - 1
    # T3 = 4z^3 - 3z
    
    T0_x = np.ones_like(x)
    T1_x = x_norm
    T2_x = 2.0 * x_norm ** 2 - 1.0
    T3_x = 4.0 * x_norm ** 3 - 3.0 * x_norm
    
    T0_v = np.ones_like(v)
    T1_v = v_norm
    T2_v = 2.0 * v_norm ** 2 - 1.0
    
    # Basis functions
    f0 = T0_x          # constant
    f1 = T1_x          # linear in x
    f2 = T1_v          # linear in v
    f3 = T2_x          # quadratic in x
    f4 = T3_x          # cubic in x (Duffing-like)
    f5 = T2_v          # quadratic in v
    f6 = T1_x * T1_v   # cross term x*v
    f7 = T2_x * T1_v   # cross term x^2*v
    f8 = np.sin(t_c)   # sinusoidal forcing
    f9 = np.cos(t_c)   # cosinusoidal forcing
    
    # Linear combination
    return (p0 * f0 + p1 * f1 + p2 * f2 + p3 * f3 + p4 * f4 +
            p5 * f5 + p6 * f6 + p7 * f7 + p8 * f8 + p9 * f9)
# EVOLVE-BLOCK-END
