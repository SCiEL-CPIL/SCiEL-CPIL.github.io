import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a coefficient-guessed basis.
    
    The basis terms are scaled and signed such that a "standard" nonlinear
    oscillator (with restoring and damping coefficients in the range of the
    training data) results in optimal parameters close to 1. This allows the
    BFGS optimizer, which starts at params=1, to converge quickly to the
    correct basin of attraction.
    
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
    
    # Estimated coefficients based on training ranges:
    # x in [-0.57, 0.53], v in [-0.17, 0.28], a in [-0.10, 0.11]
    # We assume dominant terms are linear restoring/damping and some nonlinear corrections.
    
    # 1. Linear restoring (Duffing): a ~ -k x. k ~ 0.15
    b1 = -0.15 * x
    
    # 2. Linear damping: a ~ -c v. c ~ 0.30
    b2 = -0.30 * v
    
    # 3. Cubic restoring (Duffing): a ~ -beta x^3. beta ~ 0.5
    b3 = -0.5 * x**3
    
    # 4. Cubic damping: a ~ -gamma v^3. gamma ~ 0.5
    b4 = -0.5 * v**3
    
    # 5. VDP coupling: a ~ mu x^2 v. mu ~ 0.5
    b5 = 0.5 * x**2 * v
    
    # 6. Mixed cross: a ~ delta x v^2. delta ~ 0.5
    b6 = 0.5 * x * v**2
    
    # 7. Quartic position: a ~ alpha x^4. alpha ~ 0.5
    b7 = 0.5 * x**4
    
    # 8. Quartic velocity: a ~ epsilon v^4. epsilon ~ 0.5
    b8 = 0.5 * v**4
    
    # 9. Quadratic position (bias-like): a ~ zeta x^2. zeta ~ 0.1
    b9 = 0.1 * x**2
    
    # 10. Quadratic velocity: a ~ eta v^2. eta ~ 0.1
    b10 = 0.1 * v**2
    
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
