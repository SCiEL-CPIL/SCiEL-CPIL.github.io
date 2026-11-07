import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model using Chebyshev basis for x, v
    and sinusoidal basis for time.
    
    Params:
    0: Offset
    1: Linear x (scaled)
    2: Linear v (scaled)
    3: Quadratic x (Chebyshev T2)
    4: Quadratic v (Chebyshev T2)
    5: Cubic x (Chebyshev T3)
    6: Coupling x*v
    7: Sinusoidal drive amplitude (sin)
    8: Sinusoidal drive amplitude (cos)
    9: Drive frequency
    """
    # Constants for scaling x and v to [-1, 1]
    x_min, x_max = -0.2056, 0.1749
    v_min, v_max = -0.3391, 0.3624
    
    # Calculate scaled coordinates
    # u = (2x - (xmax+xmin)) / (xmax-xmin)
    # w = (2v - (vmax+vmin)) / (vmax-vmin)
    x_range = x_max - x_min
    v_range = v_max - v_min
    
    u = (2.0 * x - (x_max + x_min)) / x_range
    w = (2.0 * v - (v_max + v_min)) / v_range
    
    # Center time to improve conditioning of frequency parameter
    t_c = t - 40.0
    
    # Extract parameters
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    omega = params[9]
    
    # Compute basis terms
    # Chebyshev Polynomials
    # T2(u) = 2u^2 - 1
    # T2(w) = 2w^2 - 1
    # T3(u) = 4u^3 - 3u
    # Cross term
    t2_u = 2.0 * u**2 - 1.0
    t2_w = 2.0 * w**2 - 1.0
    t3_u = 4.0 * u**3 - 3.0 * u
    cross = u * w
    
    # Time terms
    # To avoid issues with very large omega causing rapid oscillations and numerical noise
    # we clip omega to a reasonable range. BFGS should keep it sane, but just in case.
    # If omega is huge, sin/cos are effectively random noise.
    # A range of [-10, 10] should cover most physical frequencies for this time window.
    omega_safe = np.clip(omega, -10.0, 10.0)
    
    sin_term = np.sin(omega_safe * t_c)
    cos_term = np.cos(omega_safe * t_c)
    
    # Combine
    a = (p0 
         + p1 * u 
         + p2 * w 
         + p3 * t2_u 
         + p4 * t2_w 
         + p5 * t3_u 
         + p6 * cross 
         + p7 * sin_term 
         + p8 * cos_term)
         
    return a
