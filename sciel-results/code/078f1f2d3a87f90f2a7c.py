"""Fourier series expansion with fixed orthogonal frequencies over [0.1, 6]."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series expansion:
    f(x) = params[0] + params[1]*sin(pi*x) + params[2]*cos(pi*x)
         + params[3]*sin(2*pi*x) + params[4]*cos(2*pi*x)
         + params[5]*sin(3*pi*x) + params[6]*cos(3*pi*x)
         + params[7]*sin(4*pi*x) + params[8]*cos(4*pi*x)
         + params[9]*sin(5*pi*x)
    
    10 parameters total: 1 constant + 9 Fourier terms (4 sine + 4 cosine + 1 sine)
    Actually let's use: 1 constant + 4 sine + 5 cosine = 10
    Or better: 1 constant + pairs of sin/cos for k=1,2,3,4 (8 params) + 1 more = 10
    
    Let me use: params[0] = constant
    params[1], params[2] = sin/cos of pi*x
    params[3], params[4] = sin/cos of 2*pi*x
    params[5], params[6] = sin/cos of 3*pi*x
    params[7], params[8] = sin/cos of 4*pi*x
    params[9] = sin of 5*pi*x (extra sine term)
    """
    # Precompute the Fourier basis for efficiency
    # Domain is [0.1, 6], so x is in that range
    
    # Use frequencies that give good coverage of the domain
    # pi*x where x in [0.1, 6] gives phase in [0.1*pi, 6*pi] ~ [0.31, 18.85]
    
    result = np.empty_like(x, dtype=np.float64)
    result[:] = params[0]
    
    # Add Fourier terms
    # k=1: sin(pi*x), cos(pi*x)
    sin1 = np.sin(np.pi * x)
    cos1 = np.cos(np.pi * x)
    result += params[1] * sin1 + params[2] * cos1
    
    # k=2: sin(2*pi*x), cos(2*pi*x)
    sin2 = np.sin(2.0 * np.pi * x)
    cos2 = np.cos(2.0 * np.pi * x)
    result += params[3] * sin2 + params[4] * cos2
    
    # k=3: sin(3*pi*x), cos(3*pi*x)
    sin3 = np.sin(3.0 * np.pi * x)
    cos3 = np.cos(3.0 * np.pi * x)
    result += params[5] * sin3 + params[6] * cos3
    
    # k=4: sin(4*pi*x), cos(4*pi*x)
    sin4 = np.sin(4.0 * np.pi * x)
    cos4 = np.cos(4.0 * np.pi * x)
    result += params[7] * sin4 + params[8] * cos4
    
    # k=5: sin(5*pi*x) only (to use 10th parameter)
    sin5 = np.sin(5.0 * np.pi * x)
    result += params[9] * sin5
    
    return result
# EVOLVE-BLOCK-END
