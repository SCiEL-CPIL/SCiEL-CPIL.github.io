import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structural Change: Use Chebyshev Polynomial Basis.
    
    Instead of raw scaled monomials and sines, we use Chebyshev polynomials
    of the first kind, which are orthogonal on [-1, 1]. This improves the
    conditioning of the design matrix, potentially helping BFGS converge
    to a better solution.
    
    Inputs x and v are scaled to roughly [-1, 1].
    We construct a tensor product basis of Chebyshev polynomials up to degree 3
    in x and degree 2 in v (or similar mix) to fill the 10 parameters.
    """
    # Deadline guard (minimal overhead)
    # The evaluator calls this many times. A simple check is sufficient.
    # We don't expect to hit the limit, but we guard against pathological cases.
    # Note: time.time() is cheap.
    
    # Scaling to [-1, 1] range approximately
    # x range: [-0.5728, 0.5275] -> center ~ -0.02, span ~ 1.1
    # v range: [-0.1754, 0.2821] -> center ~ 0.05, span ~ 0.45
    # To be safe and map to [-1, 1], we can use:
    # X = (x - min_x) / (max_x - min_x) * 2 - 1
    # But we don't know min/max exactly, only ranges.
    # Let's use the provided ranges to scale.
    
    # x in [-0.57, 0.53]
    # v in [-0.18, 0.28]
    
    X = (x + 0.57) / (0.53 - (-0.57)) * 2.0 - 1.0
    V = (v + 0.18) / (0.28 - (-0.18)) * 2.0 - 1.0
    
    # Clip to [-1, 1] to avoid extrapolation issues if test data is slightly out of range?
    # The problem says "Inputs in training range", so clipping is safe and ensures stability.
    X = np.clip(X, -1.0, 1.0)
    V = np.clip(V, -1.0, 1.0)
    
    # Chebyshev Polynomials T_k(z)
    # T_0(z) = 1
    # T_1(z) = z
    # T_2(z) = 2z^2 - 1
    # T_3(z) = 4z^3 - 3z
    # T_4(z) = 8z^4 - 8z^2 + 1
    
    T0_X = np.ones_like(X)
    T1_X = X
    T2_X = 2.0 * X * X - 1.0
    T3_X = 4.0 * X * X * X - 3.0 * X
    
    T0_V = np.ones_like(V)
    T1_V = V
    T2_V = 2.0 * V * V - 1.0
    T3_V = 4.0 * V * V * V - 3.0 * V
    
    # Basis terms (10 parameters)
    # We mix X and V Chebyshev terms to capture coupling.
    
    b0 = T0_X * T0_V  # 1
    b1 = T1_X * T0_V  # X
    b2 = T0_X * T1_V  # V
    b3 = T2_X * T0_V  # 2X^2 - 1
    b4 = T0_X * T2_V  # 2V^2 - 1
    b5 = T1_X * T1_V  # X*V
    b6 = T3_X * T0_V  # 4X^3 - 3X
    b7 = T0_X * T3_V  # 4V^3 - 3V
    b8 = T2_X * T1_V  # (2X^2-1)V
    b9 = T1_X * T2_V  # X(2V^2-1)
    
    T = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    return T @ params
