import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a hybrid basis:
    - Physics-motivated terms (Duffing + damping + forcing)
    - Gaussian RBF terms for localized nonlinear corrections
    - Quadratic/cross terms for richer nonlinearity
    
    10 parameters total.
    """
    t0 = time.time()
    
    # Safety guard
    if time.time() - t0 > 140.0:
        return np.zeros_like(t, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scale inputs to O(1) for conditioning
    xs = x / 0.36
    vs = v / 0.21
    ts = t / 2.0  # assuming t is in radians, scale by ~2 for O(1)
    
    # Physics-motivated basis (6 terms)
    b0 = 1.0                          # bias
    b1 = xs                           # linear restoring
    b2 = xs * xs * xs                 # Duffing hardening (odd, no x^2 for symmetry)
    b3 = vs                           # linear damping
    b4 = np.sin(t)                    # forcing in phase
    b5 = np.cos(t)                    # forcing quadrature
    
    # RBF-like localized terms (4 terms)
    # Centers chosen within input ranges, widths scaled to domain
    # RBF in (x, v) space with different centers
    c1_x, c1_v = 0.1, 0.05
    c2_x, c2_v = -0.1, -0.05
    c3_x, c3_v = 0.0, 0.1
    c4_x, c4_v = 0.0, -0.1
    
    width = 0.25  # width parameter for Gaussian RBFs
    
    r1 = (xs - c1_x)**2 + (vs - c1_v)**2
    r2 = (xs - c2_x)**2 + (vs - c2_v)**2
    r3 = (xs - c3_x)**2 + (vs - c3_v)**2
    r4 = (xs - c4_x)**2 + (vs - c4_v)**2
    
    b6 = np.exp(-r1 / (2 * width * width))  # RBF 1
    b7 = np.exp(-r2 / (2 * width * width))  # RBF 2
    b8 = np.exp(-r3 / (2 * width * width))  # RBF 3
    b9 = np.exp(-r4 / (2 * width * width))  # RBF 4
    
    # Combine
    result = (p0 * b0 +
              p1 * b1 +
              p2 * b2 +
              p3 * b3 +
              p4 * b4 +
              p5 * b5 +
              p6 * b6 +
              p7 * b7 +
              p8 * b8 +
              p9 * b9)
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        return np.zeros_like(t, dtype=np.float64)
    
    return result
