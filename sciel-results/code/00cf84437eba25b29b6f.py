"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis combining well-conditioned Chebyshev polynomials,
    trigonometric functions, and smooth nonlinear activations.
    
    The evaluator fits the 10 params via BFGS starting from all ones,
    minimizing training MSE. We provide a rich but well-conditioned basis
    that can approximate many common hidden functions:
    
    Basis functions:
    0: T_0(xs)       (constant)
    1: T_1(xs)       (linear in scaled x)
    2: T_2(xs)       (quadratic Chebyshev)
    3: T_3(xs)       (cubic Chebyshev)
    4: T_4(xs)       (quartic Chebyshev)
    5: sin(x)        (periodic)
    6: cos(x)        (periodic)
    7: exp(-x^2)     (Gaussian bump)
    8: tanh(x)       (smooth sigmoid)
    9: x * sin(x)    (modulated basis)
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Scale x to [-1, 1] for Chebyshev stability
    xs = x / 3.0
    
    # Chebyshev polynomials T_0..T_4 (well-conditioned on [-1,1])
    b0 = np.ones_like(x)
    b1 = xs
    b2 = 2.0 * xs * xs - 1.0
    b3 = 4.0 * xs * xs * xs - 3.0 * xs
    b4 = 8.0 * xs * xs * xs * xs - 8.0 * xs * xs + 1.0
    
    # Trigonometric basis (for periodic functions)
    b5 = np.sin(x)
    b6 = np.cos(x)
    
    # Smooth nonlinear activations for local flexibility
    b7 = np.exp(-x * x)
    b8 = np.tanh(x)
    
    # Modulated basis (captures x*sin(x)-type behavior)
    b9 = x * np.sin(x)
    
    # Linear combination
    y = (params[0] * b0 +
         params[1] * b1 +
         params[2] * b2 +
         params[3] * b3 +
         params[4] * b4 +
         params[5] * b5 +
         params[6] * b6 +
         params[7] * b7 +
         params[8] * b8 +
         params[9] * b9)
         
    return y
# EVOLVE-BLOCK-END
