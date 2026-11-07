import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid RBF + polynomial model for nonlinear oscillator acceleration.
    
    Structure: a = f(x, v) = polynomial_basis + sum of Gaussian RBFs
    
    The polynomial part captures global trends (restoring, damping).
    The RBF part provides localized corrections that can capture
    non-smooth or highly nonlinear features a pure polynomial cannot.
    
    Parameters (10):
      p0: constant bias
      p1: x coefficient (linear restoring)
      p2: v coefficient (linear damping)
      p3: x^2 coefficient
      p4: v^2 coefficient
      p5: x*v coupling
      p6: RBF amplitude 1
      p7: RBF amplitude 2
      p8: RBF amplitude 3
      p9: RBF width (shared sigma)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial basis: low-order terms for global structure
    poly = p0 + p1 * x + p2 * v + p3 * x * x + p4 * v * v + p5 * x * v
    
    # RBF (Gaussian kernel) basis: localized bumps
    # Fixed centers chosen to cover the input domain
    # x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
    # Use 3 RBFs at representative (x, v) locations
    
    # RBF centers (fixed, not fitted)
    cx1, cv1 = 0.0, 0.0
    cx2, cv2 = 0.3, 0.15
    cx3, cv3 = -0.3, -0.1
    
    # Compute squared distances to each center
    dx1 = x - cx1
    dv1 = v - cv1
    r1_sq = dx1 * dx1 + dv1 * dv1
    
    dx2 = x - cx2
    dv2 = v - cv2
    r2_sq = dx2 * dx2 + dv2 * dv2
    
    dx3 = x - cx3
    dv3 = v - cv3
    r3_sq = dx3 * dx3 + dv3 * dv3
    
    # Shared width parameter p9; use abs to keep sigma positive
    sigma = np.abs(p9) + 1e-6
    sigma2 = 2.0 * sigma * sigma
    
    # Gaussian RBFs
    rbf1 = np.exp(-r1_sq / sigma2)
    rbf2 = np.exp(-r2_sq / sigma2)
    rbf3 = np.exp(-r3_sq / sigma2)
    
    result = poly + p6 * rbf1 + p7 * rbf2 + p8 * rbf3
    
    return result
