import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a physics-informed Duffing-van der Pol
    structure with a Chebyshev-like normalized basis for better conditioning.
    
    The structure is: a = f(x) + g(v) + h(x,v)
    where f, g, h are polynomial terms motivated by oscillator physics.
    
    We use normalized coordinates to keep the basis well-conditioned for BFGS
    starting from all-ones.
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Normalize to roughly [-1, 1] for better conditioning
    # x range: [-0.5728, 0.5275]
    x_center = (-0.5728 + 0.5275) / 2.0
    x_radius = (0.5275 - (-0.5728)) / 2.0
    
    # v range: [-0.1754, 0.2821]
    v_center = (-0.1754 + 0.2821) / 2.0
    v_radius = (0.2821 - (-0.1754)) / 2.0
    
    X = (x - x_center) / x_radius
    V = (v - v_center) / v_radius
    
    # Clip to avoid numerical issues
    X = np.clip(X, -1.0, 1.0)
    V = np.clip(V, -1.0, 1.0)
    
    # Physics-informed Duffing-van der Pol structure
    # Restoring force: linear + cubic (Duffing)
    # Damping: linear + cubic (van der Pol)
    # Cross terms for coupling
    
    # Position-dependent terms
    b1 = X              # Linear restoring force
    b2 = X**3           # Cubic (Duffing)
    b3 = X**2           # Quadratic (asymmetry)
    
    # Velocity-dependent terms
    b4 = V              # Linear damping
    b5 = V**3           # Cubic (van der Pol)
    b6 = V**2           # Quadratic damping
    
    # Cross/coupling terms
    b7 = X * V**2       # x * v^2 coupling
    b8 = V * X**2       # v * x^2 coupling
    b9 = X**2 * V**2    # x^2 * v^2 coupling
    
    # Constant
    b10 = np.ones_like(X)
    
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
