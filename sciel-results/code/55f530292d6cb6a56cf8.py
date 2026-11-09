"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator using a monomial polynomial basis
    mapped to the training domain.
    
    This approach uses a monomial basis up to degree 3 for both position and velocity,
    providing a well-conditioned basis that helps BFGS converge reliably from
    an initial guess of all ones. The structure captures both restoring force
    (position-dependent) and damping (velocity-dependent) terms with cross-coupling.
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Map training ranges to [-1, 1] for polynomial expansion
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
    
    # Compute monomial basis terms
    # Position dependent terms (Restoring force)
    b1 = X              # Linear
    b2 = X**3           # Cubic (Duffing)
    b3 = X**2           # Quadratic (Asymmetry)
    
    # Velocity dependent terms (Damping)
    b4 = V              # Linear
    b5 = V**3           # Cubic
    b6 = V**2           # Quadratic
    
    # Cross terms (Coupling)
    b7 = X * V
    b8 = X**2 * V
    b9 = X * V**2
    
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
         
    # Ensure finite values to prevent optimizer failure
    a = np.where(np.isfinite(a), a, 0.0)
    
    return a
# EVOLVE-BLOCK-END
