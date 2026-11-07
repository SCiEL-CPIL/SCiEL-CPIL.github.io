"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Extract parameters for numerator and denominator
    # 4/4 Padé approximation: 5 coeffs for P(x), 5 coeffs for Q(x)
    p = params[0:5]
    q = params[5:10]
    
    # Compute powers of x
    x0 = np.ones_like(x)
    x1 = x
    x2 = x**2
    x3 = x**3
    x4 = x**4
    
    # Numerator P(x)
    P = p[0]*x0 + p[1]*x1 + p[2]*x2 + p[3]*x3 + p[4]*x4
    
    # Denominator Q(x)
    Q = q[0]*x0 + q[1]*x1 + q[2]*x2 + q[3]*x3 + q[4]*x4
    
    # Safety: prevent division by zero or very small values
    # If Q is close to 0, replace with a small epsilon with the same sign
    # to keep the function finite and differentiable (mostly)
    eps = 1e-8
    Q_safe = np.where(np.abs(Q) < eps, np.sign(Q) * eps, Q)
    # Handle case where Q is exactly 0 (sign is 0), replace with eps
    Q_safe = np.where(Q_safe == 0, eps, Q_safe)
    
    return P / Q_safe
# EVOLVE-BLOCK-END
