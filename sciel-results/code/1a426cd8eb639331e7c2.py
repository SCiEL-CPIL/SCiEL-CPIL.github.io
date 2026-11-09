import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a sparse Chebyshev polynomial basis
    focused on physically relevant terms. Uses fewer, well-motivated basis
    functions to improve BFGS convergence from the all-ones initial guess.
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Map training ranges to [-1, 1] for Chebyshev expansion
    # x range: [-0.5728, 0.5275]
    x_min, x_max = -0.5728, 0.5275
    x_center = (x_min + x_max) / 2.0
    x_radius = (x_max - x_min) / 2.0
    
    # v range: [-0.1754, 0.2821]
    v_min, v_max = -0.1754, 0.2821
    v_center = (v_min + v_max) / 2.0
    v_radius = (v_max - v_min) / 2.0
    
    # Normalize to [-1, 1]
    X = (x - x_center) / x_radius
    V = (v - v_center) / v_radius
    
    # Soft-clip to avoid hard discontinuities in gradients
    # Use tanh-based soft clipping for smoothness
    X_soft = np.tanh(X)
    V_soft = np.tanh(V)
    
    # Compute Chebyshev polynomials on soft-clipped values
    # T_0 = 1
    # T_1 = X
    # T_2 = 2X^2 - 1
    # T_3 = 4X^3 - 3X
    # T_4 = 8X^4 - 8X^2 + 1
    
    T0 = np.ones_like(X_soft)
    T1_X = X_soft
    T2_X = 2.0 * X_soft**2 - 1.0
    T3_X = 4.0 * X_soft**3 - 3.0 * X_soft
    T4_X = 8.0 * X_soft**4 - 8.0 * X_soft**2 + 1.0
    
    T1_V = V_soft
    T2_V = 2.0 * V_soft**2 - 1.0
    T3_V = 4.0 * V_soft**3 - 3.0 * V_soft
    
    # Sparse, physics-motivated basis:
    # b1: Linear restoring force (T1_X ~ x)
    # b2: Cubic Duffing nonlinearity (T3_X ~ x^3)
    # b3: Linear damping (T1_V ~ v)
    # b4: Van der Pol coupling (T1_X * T1_V ~ x*v)
    # b5: Quadratic position asymmetry (T2_X ~ x^2)
    # b6: Quadratic velocity damping (T2_V ~ v^2)
    # b7: Higher-order position (T4_X ~ x^4)
    # b8: Position-velocity cross (T2_X * T1_V ~ x^2*v)
    # b9: Velocity-cubic (T3_V ~ v^3)
    # b10: Constant offset
    
    b1 = T1_X
    b2 = T3_X
    b3 = T1_V
    b4 = T1_X * T1_V
    b5 = T2_X
    b6 = T2_V
    b7 = T4_X
    b8 = T2_X * T1_V
    b9 = T3_V
    b10 = T0
    
    a = (p[0] * b1 +
         p[1] * b2 +
         p[2] * b3 +
         p[3] * b4 +
         p[4] * b5 +
         p[5] * b6 +
         p[6] * b7 +
         p[7] * b8 +
         p[8] * b9 +
         p[9] * b10)
         
    return a
