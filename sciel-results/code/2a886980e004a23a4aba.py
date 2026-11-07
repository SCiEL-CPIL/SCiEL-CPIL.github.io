"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Product of 1D Fourier-like trigonometric terms to model periodic/oscillatory
    target functions. The 10 parameters are amplitudes for a basis of products
    of cos/sin of linear combinations of x and y.
    
    Basis:
    p0 * 1
    p1 * cos(x)
    p2 * sin(x)
    p3 * cos(y)
    p4 * sin(y)
    p5 * cos(x+y)
    p6 * sin(x+y)
    p7 * cos(x-y)
    p8 * sin(x-y)
    p9 * cos(2*x) * cos(2*y)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute trigonometric components
    cos_x = np.cos(x)
    sin_x = np.sin(x)
    cos_y = np.cos(y)
    sin_y = np.sin(y)
    
    sum_xy = x + y
    diff_xy = x - y
    
    cos_sum = np.cos(sum_xy)
    sin_sum = np.sin(sum_xy)
    cos_diff = np.cos(diff_xy)
    sin_diff = np.sin(diff_xy)
    
    cos_2x = np.cos(2 * x)
    cos_2y = np.cos(2 * y)
    
    # Build the basis as a weighted sum
    basis = (p0 * np.ones_like(x) +
             p1 * cos_x +
             p2 * sin_x +
             p3 * cos_y +
             p4 * sin_y +
             p5 * cos_sum +
             p6 * sin_sum +
             p7 * cos_diff +
             p8 * sin_diff +
             p9 * cos_2x * cos_2y)
    
    return basis
# EVOLVE-BLOCK-END
