import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a rich basis of 10 functions to approximate
    the unknown target function via BFGS-fitted parameters.
    
    The basis combines:
    - Polynomial terms (x, x^2, x^3) for smooth local behavior
    - Trigonometric terms (sin(pi*x/2), cos(pi*x/2)) for oscillatory behavior
    - Exponential terms (exp(x/2), exp(-x/2)) for growth/decay
    - Sigmoid-like term (tanh(x)) for step-function behavior
    - Constant term (1)
    
    This diverse basis can capture a wider variety of functional forms than
    pure polynomials alone.
    """
    # Safety: ensure x is 1D array
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Basis functions
    # 0: constant
    b0 = np.ones_like(x)
    # 1: linear
    b1 = x
    # 2: quadratic
    b2 = x ** 2
    # 3: cubic
    b3 = x ** 3
    # 4: sine (period 4, matches domain [-2,2] nicely)
    b4 = np.sin(np.pi * x / 2.0)
    # 5: cosine
    b5 = np.cos(np.pi * x / 2.0)
    # 6: exponential growth (scaled to avoid overflow)
    b6 = np.exp(0.5 * x)
    # 7: exponential decay
    b7 = np.exp(-0.5 * x)
    # 8: tanh (smooth step)
    b8 = np.tanh(x)
    # 9: Gaussian-like bump (peaked at 0)
    b9 = np.exp(-x ** 2)
    
    # Build basis matrix (N, 10)
    basis = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Linear combination with parameters
    return basis @ params
