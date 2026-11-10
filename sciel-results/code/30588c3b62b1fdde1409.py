"""Symbolic regression with a localized Gaussian basis and polynomial backbone."""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: sum of 3 Gaussian bumps + polynomial backbone (degree 3).
    
    Parameters (10 total):
      p0: amplitude of Gaussian 1
      p1: mean of Gaussian 1 (clipped to [-2, 2])
      p2: width of Gaussian 1 (clipped to [0.1, 3.0])
      p3: amplitude of Gaussian 2
      p4: mean of Gaussian 2 (clipped to [-2, 2])
      p5: width of Gaussian 2 (clipped to [0.1, 3.0])
      p6: amplitude of Gaussian 3
      p7: mean of Gaussian 3 (clipped to [-2, 2])
      p8: width of Gaussian 3 (clipped to [0.1, 3.0])
      p9: coefficient for x (linear backbone)
    
    The Gaussian basis allows adaptive placement of localized features,
    while the linear term provides a global trend. This structure can
    approximate multi-modal, sharp, or discontinuous functions better
    than a purely global polynomial or trigonometric basis.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Extract and clip parameters for numerical safety
    a1, m1, w1 = p[0], p[1], p[2]
    a2, m2, w2 = p[3], p[4], p[5]
    a3, m3, w3 = p[6], p[7], p[8]
    c_lin = p[9]
    
    # Clip means to domain and widths to safe positive range
    m1 = np.clip(m1, -2.0, 2.0)
    m2 = np.clip(m2, -2.0, 2.0)
    m3 = np.clip(m3, -2.0, 2.0)
    w1 = np.clip(w1, 0.1, 3.0)
    w2 = np.clip(w2, 0.1, 3.0)
    w3 = np.clip(w3, 0.1, 3.0)
    
    # Compute Gaussian terms
    # Using exp(-w * (x - m)^2) form
    g1 = np.exp(-w1 * (x - m1) ** 2)
    g2 = np.exp(-w2 * (x - m2) ** 2)
    g3 = np.exp(-w3 * (x - m3) ** 2)
    
    # Linear backbone
    lin = c_lin * x
    
    # Combine
    result = a1 * g1 + a2 * g2 + a3 * g3 + lin
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
