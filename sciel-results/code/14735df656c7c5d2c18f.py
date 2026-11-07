"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x to [-1, 1] range to improve conditioning of polynomial terms
    xs = x / 2.0
    
    # Precompute basis functions
    # Polynomial terms (scaled)
    p1 = np.ones_like(x)
    p2 = xs
    p3 = xs * xs
    p4 = xs * xs * xs
    
    # Trigonometric terms
    t1 = np.sin(x)
    t2 = np.cos(x)
    t3 = np.sin(2.0 * x)
    t4 = np.cos(2.0 * x)
    
    # Non-linear smooth terms
    n1 = np.tanh(x)
    n2 = np.exp(-x * x)
    
    # Combine using parameters
    # Order: p1..p4, t1..t4, n1, n2
    pred = (params[0] * p1 + 
            params[1] * p2 + 
            params[2] * p3 + 
            params[3] * p4 + 
            params[4] * t1 + 
            params[5] * t2 + 
            params[6] * t3 + 
            params[7] * t4 + 
            params[8] * n1 + 
            params[9] * n2)
            
    return pred
# EVOLVE-BLOCK-END
