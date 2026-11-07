import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a robust basis of Chebyshev polynomials, 
    higher-frequency trigonometric functions, and a non-linear term.
    
    Strategy:
    - Use Chebyshev polynomials T_0..T_4 on normalized x in [-1, 1] for 
      better conditioning than monomials.
    - Use Sin/Cos with frequencies 1, 2, 3 to capture higher-frequency 
      oscillations.
    - Use a Tanh term to capture saturation/step-like behaviors.
    """
    # Normalize x to [-1, 1] for Chebyshev and better trig scaling
    xn = x / 2.0
    
    # 1. Chebyshev Polynomials (Degree 0 to 4)
    # T_0(x) = 1
    # T_1(x) = x
    # T_2(x) = 2x^2 - 1
    # T_3(x) = 4x^3 - 3x
    # T_4(x) = 8x^4 - 8x^2 + 1
    
    t0 = params[0] * 1.0
    t1 = params[1] * xn
    t2 = params[2] * (2.0 * xn**2 - 1.0)
    t3 = params[3] * (4.0 * xn**3 - 3.0 * xn)
    t4 = params[4] * (8.0 * xn**4 - 8.0 * xn**2 + 1.0)
    
    # 2. Trigonometric Functions (Frequencies 1, 2, 3)
    # Using original x for trig to cover [-2, 2] range naturally.
    # Frequencies 1, 2, 3 allow capturing periods of 2pi, pi, 2pi/3.
    s1 = params[5] * np.sin(x)
    c1 = params[6] * np.cos(x)
    s2 = params[7] * np.sin(2.0 * x)
    c2 = params[8] * np.cos(2.0 * x)
    
    # 3. Non-linear Saturation / Step Approximation
    # Tanh is smooth and bounded, good for step-like or sigmoidal features.
    # Alternatively, a Gaussian bump. Tanh is often more robust for 
    # "unknown" functions that might have a transition.
    n1 = params[9] * np.tanh(x)
    
    return t0 + t1 + t2 + t3 + t4 + s1 + c1 + s2 + c2 + n1
