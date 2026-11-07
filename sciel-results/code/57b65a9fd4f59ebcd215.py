"""
STRUCTURAL CHANGE: Product-based feature expansion with Chebyshev polynomials
multiplied by nonlinear activations.

The key insight: instead of a fixed linear basis or a simple neural network
layer, we create 10 basis functions that are PRODUCTS of:
- Low-degree Chebyshev polynomials (global stability, orthogonal on [-1,1])
- Nonlinear activations (tanh, sin) to capture non-periodic nonlinearities

This gives BFGS a convex problem in params (linear combination) while the
basis itself has rich nonlinear structure. The Chebyshev polynomials provide
numerical stability compared to raw powers, and the product with activations
captures smooth nonlinear trends without the locality constraints of RBFs.

Basis functions (10 total):
- T_0(x') * 1          (constant)
- T_1(x') * 1          (linear)
- T_2(x') * 1          (quadratic)
- T_3(x') * 1          (cubic)
- T_4(x') * 1          (quartic)
- T_1(x') * tanh(x')   (linear * tanh)
- T_2(x') * tanh(x')   (quadratic * tanh)
- T_1(x') * sin(x')    (linear * sin)
- T_2(x') * sin(x')    (quadratic * sin)
- T_3(x') * tanh(2*x') (cubic * tanh, higher frequency)

where x' = x/3 maps [-3,3] to [-1,1] for Chebyshev stability.

All 10 parameters are linear in the output, so BFGS solves a convex problem.
"""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Map x from [-3, 3] to [-1, 1] for Chebyshev stability
    xp = x / 3.0
    
    # Compute Chebyshev polynomials T_0 through T_4
    T0 = np.ones_like(xp)
    T1 = xp
    T2 = 2.0 * xp * xp - 1.0
    T3 = 4.0 * xp**3 - 3.0 * xp
    T4 = 8.0 * xp**4 - 8.0 * xp**2 + 1.0
    
    # Nonlinear activations
    tanh_x = np.tanh(xp)
    sin_x = np.sin(xp)
    tanh_2x = np.tanh(2.0 * xp)
    
    # Product basis functions
    b0 = T0 * 1.0              # constant
    b1 = T1 * 1.0              # linear
    b2 = T2 * 1.0              # quadratic
    b3 = T3 * 1.0              # cubic
    b4 = T4 * 1.0              # quartic
    b5 = T1 * tanh_x           # linear * tanh
    b6 = T2 * tanh_x           # quadratic * tanh
    b7 = T1 * sin_x            # linear * sin
    b8 = T2 * sin_x            # quadratic * sin
    b9 = T3 * tanh_2x          # cubic * tanh(2x)
    
    # Linear combination
    y = (p[0] * b0 +
         p[1] * b1 +
         p[2] * b2 +
         p[3] * b3 +
         p[4] * b4 +
         p[5] * b5 +
         p[6] * b6 +
         p[7] * b7 +
         p[8] * b8 +
         p[9] * b9)
    
    # Ensure finite output
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
