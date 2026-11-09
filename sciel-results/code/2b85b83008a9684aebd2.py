"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of diverse basis functions.
    The basis functions are scaled and centered to improve the conditioning of the
    optimization problem for BFGS.
    """
    # Ensure x is a 1D array of floats
    x = np.asarray(x, dtype=np.float64)
    
    # Define 10 basis functions. 
    # We scale them to have roughly zero mean and unit variance over [-2, 2] 
    # to help BFGS converge from the initial guess of all ones.
    
    # 1. Constant
    b0 = np.ones_like(x)
    
    # 2. Linear
    b1 = x
    
    # 3. Quadratic (centered to reduce correlation with constant)
    # Mean of x^2 on [-2,2] is 4/3. 
    b2 = x**2 - 4.0/3.0
    
    # 4. Cubic (centered)
    # Mean of x^3 is 0. Mean of x^5 is 16/5=3.2. 
    # Let's just use x^3, it's odd, so mean is 0.
    b3 = x**3
    
    # 5. Sine
    b4 = np.sin(x)
    
    # 6. Cosine (centered)
    # Mean of cos(x) on [-2,2] is sin(2)/2. sin(2) ~ 0.909. Mean ~ 0.45.
    b5 = np.cos(x) - 0.45
    
    # 7. Exponential growth (scaled/centered)
    # e^x on [-2,2] ranges [0.13, 7.39]. Mean is (e^2-e^-2)/4 ~ 1.8.
    b6 = np.exp(x) - 1.8
    
    # 8. Exponential decay (scaled/centered)
    b7 = np.exp(-x) - 1.8
    
    # 9. Hyperbolic tangent (odd, mean 0)
    b8 = np.tanh(x)
    
    # 10. Lorentzian / Rational (even, peaked at 0)
    # 1/(1+x^2) on [-2,2]. At 0: 1. At 2: 0.2. Mean is roughly 0.4.
    b9 = 1.0 / (1.0 + x**2) - 0.4

    # Combine linearly
    # params is an array of length 10
    return (
        params[0] * b0 +
        params[1] * b1 +
        params[2] * b2 +
        params[3] * b3 +
        params[4] * b4 +
        params[5] * b5 +
        params[6] * b6 +
        params[7] * b7 +
        params[8] * b8 +
        params[9] * b9
    )
# EVOLVE-BLOCK-END
