"""Symbolic regression: hybrid Chebyshev + Fourier basis, linear in params."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form prediction using a hybrid basis:
    - Chebyshev polynomials T_0..T_5 on normalized x (captures smooth trends)
    - Fourier terms sin/cos of x and 2x (captures periodic components)
    - A constant term
    
    The model is linear in params, so BFGS converges to the global optimum
    quickly (convex least-squares subproblem).
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Normalize x to [-1, 1] for better Chebyshev conditioning
    # Original range is [-3, 3], so divide by 3
    xn = x / 3.0
    
    # Chebyshev polynomials T_0 through T_5
    # T_0(x) = 1
    # T_1(x) = x
    # T_2(x) = 2x^2 - 1
    # T_3(x) = 4x^3 - 3x
    # T_4(x) = 8x^4 - 8x^2 + 1
    # T_5(x) = 16x^5 - 20x^3 + 5x
    t0 = np.ones_like(xn)
    t1 = xn
    t2 = 2.0 * xn**2 - 1.0
    t3 = 4.0 * xn**3 - 3.0 * xn
    t4 = 8.0 * xn**4 - 8.0 * xn**2 + 1.0
    t5 = 16.0 * xn**5 - 20.0 * xn**3 + 5.0 * xn
    
    # Fourier terms on original x (periodic components)
    f1s = np.sin(x)
    f1c = np.cos(x)
    f2s = np.sin(2.0 * x)
    f2c = np.cos(2.0 * x)
    
    # 10 basis functions total
    # Indices 0-5: Chebyshev T_0..T_5
    # Indices 6-9: Fourier sin(x), cos(x), sin(2x), cos(2x)
    basis = np.column_stack([t0, t1, t2, t3, t4, t5, f1s, f1c, f2s, f2c])
    
    return basis @ p
# EVOLVE-BLOCK-END
