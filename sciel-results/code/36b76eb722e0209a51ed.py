"""
Strategy: Hybrid Basis Function Approximation.

Uses a fixed basis of 10 functions that is linear in the parameters.
This transforms the optimization problem into a convex linear least-squares problem,
which BFGS solves robustly and quickly.

The basis includes:
1. Low-order polynomials (1 to x^4) for general smooth trends.
2. Sine and Cosine terms for potential periodicity.
3. A Gaussian term for local features.

Inputs x are in [-2, 2].
"""
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure x is a 1D array for consistent shape handling
    x = np.asarray(x).flatten()
    
    # Build the basis matrix (N x 10)
    # 1. Constant
    b0 = np.ones_like(x)
    
    # 2-5. Polynomials
    b1 = x
    b2 = x**2
    b3 = x**3
    b4 = x**4
    
    # 6-9. Trigonometric functions
    # Frequencies chosen to cover the domain [-2, 2]
    # pi*x/2 covers one full period over [-2, 2]
    # pi*x covers two full periods over [-2, 2]
    b5 = np.sin(np.pi * x / 2.0)
    b6 = np.cos(np.pi * x / 2.0)
    b7 = np.sin(np.pi * x)
    b8 = np.cos(np.pi * x)
    
    # 10. Gaussian bump centered at 0
    # Width chosen to be local but not too narrow
    b9 = np.exp(-x**2)
    
    # Stack into a matrix (N, 10)
    basis = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    # Compute prediction: basis @ params
    # params is shape (10,)
    pred = basis.dot(params)
    
    return pred
