"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator using Chebyshev polynomial basis
    for state variables and sinusoidal basis for time.
    """
    # Scale inputs to [-1, 1] for stable Chebyshev approximation
    # t range: [30, 50]
    t_scaled = (t - 40.0) / 10.0
    
    # x range: [-0.2056, 0.1749]
    x_mid = (-0.2056 + 0.1749) / 2.0
    x_half = (0.1749 - (-0.2056)) / 2.0
    x_scaled = (x - x_mid) / x_half
    
    # v range: [-0.3391, 0.3624]
    v_mid = (-0.3391 + 0.3624) / 2.0
    v_half = (0.3624 - (-0.3391)) / 2.0
    v_scaled = (v - v_mid) / v_half

    # Chebyshev polynomials of degree 1, 2, 3
    # T_1(z) = z
    # T_2(z) = 2z^2 - 1
    # T_3(z) = 4z^3 - 3z
    
    T1_x = x_scaled
    T2_x = 2.0 * x_scaled**2 - 1.0
    T3_x = 4.0 * x_scaled**3 - 3.0 * x_scaled
    
    T1_v = v_scaled
    T2_v = 2.0 * v_scaled**2 - 1.0
    T3_v = 4.0 * v_scaled**3 - 3.0 * v_scaled
    
    # Sinusoidal time terms
    sin_t = np.sin(t_scaled)
    cos_t = np.cos(t_scaled)
    
    # Coupling term
    coupling = x_scaled * v_scaled
    
    # Combine with parameters
    # p0: offset
    # p1: linear x
    # p2: linear v
    # p3: quadratic x
    # p4: quadratic v
    # p5: cubic x
    # p6: cubic v
    # p7: coupling
    # p8: sin(t)
    # p9: cos(t)
    
    return (params[0] + 
            params[1] * T1_x + 
            params[2] * T1_v + 
            params[3] * T2_x + 
            params[4] * T2_v + 
            params[5] * T3_x + 
            params[6] * T3_v + 
            params[7] * coupling + 
            params[8] * sin_t + 
            params[9] * cos_t)
# EVOLVE-BLOCK-END
