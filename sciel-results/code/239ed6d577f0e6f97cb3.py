"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate: a truncated Fourier series on [-3, 3].
    
    The basis functions are:
      p0 * 1
      p1 * cos(pi*x/3)
      p2 * sin(pi*x/3)
      p3 * cos(2*pi*x/3)
      p4 * sin(2*pi*x/3)
      p5 * cos(3*pi*x/3)
      p6 * sin(3*pi*x/3)
      p7 * cos(4*pi*x/3)
      p8 * sin(4*pi*x/3)
      p9 * cos(5*pi*x/3)
    
    This gives 10 parameters, matching the required length.
    The Fourier basis is well-suited for smooth periodic-like functions
    on a bounded interval and provides good approximation properties.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Precompute the base argument
    arg = np.pi * x / 3.0
    
    # Start with the constant term
    y = np.full_like(x, params[0], dtype=np.float64)
    
    # Add Fourier terms
    y += params[1] * np.cos(arg)
    y += params[2] * np.sin(arg)
    
    y += params[3] * np.cos(2.0 * arg)
    y += params[4] * np.sin(2.0 * arg)
    
    y += params[5] * np.cos(3.0 * arg)
    y += params[6] * np.sin(3.0 * arg)
    
    y += params[7] * np.cos(4.0 * arg)
    y += params[8] * np.sin(4.0 * arg)
    
    y += params[9] * np.cos(5.0 * arg)
    
    return y
# EVOLVE-BLOCK-END
