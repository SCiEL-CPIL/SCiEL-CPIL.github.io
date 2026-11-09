import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a Chebyshev polynomial basis.
    Chebyshev polynomials provide a well-conditioned basis for approximation
    over a bounded interval, which helps the BFGS optimizer converge
    reliably from an initial guess of all ones.
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
    
    # Clip to [-1, 1] to avoid numerical issues if test data is slightly out of range
    # Though the task says "in-distribution", slight clipping is safe for Chebyshev
    X = np.clip(X, -1.0, 1.0)
    V = np.clip(V, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_n
    # T_0 = 1
    # T_1 = X
    # T_2 = 2X^2 - 1
    # T_3 = 4X^3 - 3X
    
    T0_X = np.ones_like(X)
    T1_X = X
    T2_X = 2.0 * X**2 - 1.0
    T3_X = 4.0 * X**3 - 3.0 * X
    
    T0_V = np.ones_like(V)
    T1_V = V
    T2_V = 2.0 * V**2 - 1.0
    T3_V = 4.0 * V**3 - 3.0 * V
    
    # Construct basis terms
    # We mix position and velocity terms.
    # A typical oscillator is a = f(x) + g(v).
    # We allocate parameters to these components.
    
    # Position dependent terms (Restoring force)
    b1 = T1_X      # Linear
    b2 = T3_X      # Cubic (Duffing)
    b3 = T2_X      # Quadratic (Asymmetry)
    
    # Velocity dependent terms (Damping)
    b4 = T1_V      # Linear
    b5 = T3_V      # Cubic
    b6 = T2_V      # Quadratic
    
    # Cross terms (Coupling)
    b7 = T1_X * T1_V
    b8 = T2_X * T1_V
    b9 = T1_X * T2_V
    
    # Constant
    b10 = T0_X     # or T0_V, same thing
    
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
