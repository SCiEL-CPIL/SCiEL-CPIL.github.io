import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a scaled monomial basis derived from
    Duffing and Van der Pol oscillator physics.
    
    The model assumes a potential well structure (Duffing: -x - x^3) and
    a damping term that depends on position (Van der Pol: v(1-x^2)).
    We also include higher-order terms to capture deviations from the
    canonical forms.
    
    Inputs are scaled to be order 1 to improve the conditioning of the Hessian
    for the BFGS optimizer, allowing it to converge reliably from an initial
    guess of all ones.
    
    Inputs:
      x: position
      v: velocity
      params: array of 10 coefficients
      
    Returns:
      a: acceleration
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Scaling factors based on training data ranges provided in the task description
    # x range: [-0.5728, 0.5275] -> max abs ~ 0.6
    # v range: [-0.1754, 0.2821] -> max abs ~ 0.3
    S_x = 0.6
    S_v = 0.3
    
    # Scaled inputs
    X = x / S_x
    V = v / S_v
    
    # Precompute powers of scaled inputs
    X2 = X * X
    X3 = X2 * X
    X4 = X2 * X2
    X5 = X4 * X
    
    V2 = V * V
    V3 = V2 * V
    V5 = V3 * V2
    
    # Define basis terms
    # Restoring (Duffing-like)
    b1 = X          # Linear
    b2 = X3         # Cubic
    b3 = X5         # Quintic
    
    # Damping (Van der Pol-like)
    b4 = V          # Linear velocity
    b5 = X2 * V     # VDP coupling
    b6 = X4 * V     # Higher VDP coupling
    
    # Nonlinear damping
    b7 = V3         # Cubic velocity
    b8 = V5         # Quintic velocity
    
    # Cross terms
    b9 = X * V      # Linear cross
    b10 = X * V3    # Mixed cross
    
    # Combine
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
