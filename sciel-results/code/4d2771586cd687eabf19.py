"""
Structural change: Product-of-terms algebraic basis.
Uses a combination of global powers x^k and local powers (x-c)^k
to create a flexible, non-periodic basis that can represent
rational-like or multi-scale behaviors. The model is linear in
the 10 parameters, ensuring BFGS stability.
"""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Product-of-terms basis on [0.1, 6] with 10 parameters.
    
    Basis functions:
      f0 = 1
      f1 = x
      f2 = x^2
      f3 = x^3
      f4 = (x - 1.0)^2
      f5 = (x - 2.0)^2
      f6 = (x - 3.0)^2
      f7 = (x - 4.0)^2
      f8 = (x - 5.0)^2
      f9 = (x - 3.0)^3
    
    This combines global polynomial behavior with local quadratic
    and cubic terms centered at distinct points in the domain,
    providing flexibility for non-periodic smooth functions.
    """
    n = len(x)
    basis = np.ones((n, 10))
    
    # Global powers
    basis[:, 1] = x
    basis[:, 2] = x ** 2
    basis[:, 3] = x ** 3
    
    # Local quadratic terms at distinct centers
    centers = [1.0, 2.0, 3.0, 4.0, 5.0]
    for i, c in enumerate(centers):
        basis[:, 4 + i] = (x - c) ** 2
    
    # One local cubic term at the center of the domain
    basis[:, 9] = (x - 3.0) ** 3
    
    return basis @ params
