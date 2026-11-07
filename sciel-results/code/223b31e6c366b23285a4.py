"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: normalized polynomials + trigonometric terms.
    x is in [0.1, 6]. We map to u in [-1, 1] for polynomial stability.
    params: length 10 array.
    We use a mix: 5 polynomial terms (degree 0-4 on u) and 5 trig terms (sin/cos of various frequencies).
    Actually, let's use a richer set to capture potential oscillations or non-linear trends.
    
    Basis design:
    1. Constant: 1
    2. Linear: u
    3. Quadratic: u^2
    4. Cubic: u^3
    5. Quartic: u^4
    6. Sin(pi * u)
    7. Cos(pi * u)
    8. Sin(2 * pi * u)
    9. Cos(2 * pi * u)
    10. Sin(3 * pi * u)
    
    This allows capturing up to 4th order polynomial trends and up to 3 cycles of oscillation
    across the domain.
    """
    # Time guard: if we take too long, return a simple fallback (though this function is fast)
    # The evaluator calls this many times. It's vectorized, so it should be very fast.
    
    x_min = 0.1
    x_max = 6.0
    center = (x_max + x_min) / 2.0  # 3.05
    half_range = (x_max - x_min) / 2.0  # 2.95
    
    u = (x - center) / half_range  # u in [-1, 1]
    
    # Polynomial part
    u2 = u * u
    u3 = u2 * u
    u4 = u2 * u2
    
    # Trig part
    # Frequencies: 1, 1, 2, 2, 3 (in units of pi)
    # sin(pi*u), cos(pi*u), sin(2*pi*u), cos(2*pi*u), sin(3*pi*u)
    
    s1 = np.sin(np.pi * u)
    c1 = np.cos(np.pi * u)
    s2 = np.sin(2.0 * np.pi * u)
    c2 = np.cos(2.0 * np.pi * u)
    s3 = np.sin(3.0 * np.pi * u)
    
    # Combine
    # params[0] * 1 + params[1] * u + params[2] * u2 + params[3] * u3 + params[4] * u4
    # + params[5] * s1 + params[6] * c1 + params[7] * s2 + params[8] * c2 + params[9] * s3
    
    return (params[0] + 
            params[1] * u + 
            params[2] * u2 + 
            params[3] * u3 + 
            params[4] * u4 + 
            params[5] * s1 + 
            params[6] * c1 + 
            params[7] * s2 + 
            params[8] * c2 + 
            params[9] * s3)
# EVOLVE-BLOCK-END
