"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Trigonometric basis (Fourier-like)
    # This approach assumes the function can be approximated by low-frequency
    # sine and cosine waves. This is distinct from polynomial regression
    # and often handles smooth, bounded, or periodic-like functions better
    # without the boundary ill-conditioning of high-degree polynomials.
    
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Basis terms
    t0 = 1.0
    t1 = np.sin(x)
    t2 = np.cos(x)
    t3 = np.sin(y)
    t4 = np.cos(y)
    t5 = np.sin(x + y)
    t6 = np.cos(x + y)
    t7 = np.sin(x - y)
    t8 = np.cos(x - y)
    t9 = x * y  # Adding a simple interaction term to capture non-separable structure
    
    return (p0 * t0 + p1 * t1 + p2 * t2 + 
            p3 * t3 + p4 * t4 + 
            p5 * t5 + p6 * t6 + 
            p7 * t7 + p8 * t8 + 
            p9 * t9)
# EVOLVE-BLOCK-END
