"""Combined symbolic regression model for a nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a driven nonlinear oscillator.
    
    This model combines the structural physics of a Duffing-Van der Pol oscillator
    (from the current program) with the numerical conditioning strategy of input
    normalization (from inspiration 2/3) and adds a cubic velocity term for
    saturation (from inspiration 1/strategy).
    
    Inputs:
      t: time [30, 50]
      x: position [-0.2056, 0.1749]
      v: velocity [-0.3391, 0.3624]
      
    Model:
      We normalize x and v to be O(1) to improve the conditioning of the Hessian
      for BFGS, which helps the optimizer converge to a better minimum.
      
      xn = x / 0.2
      vn = v / 0.35
      
      a = p0*sin(t) + p1*cos(t)       (Periodic drive)
        + p2*xn + p3*xn^3             (Duffing stiffness: linear + cubic)
        + p4*vn + p5*vn^3             (Damping: linear + cubic saturation)
        + p6*xn^2*vn                  (Van der Pol nonlinear damping)
        + p7*xn*vn                    (Cross-coupling)
        + p8*xn^2                     (Quadratic position term)
        + p9                          (Constant offset)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to improve numerical conditioning for BFGS
    # x range approx [-0.2, 0.17], scale by 0.2 -> approx [-1, 0.9]
    # v range approx [-0.34, 0.36], scale by 0.35 -> approx [-1, 1]
    xn = x * 5.0  # 1/0.2
    vn = v * (1.0 / 0.35)
    
    # Precompute powers
    xn2 = xn * xn
    xn3 = xn2 * xn
    vn2 = vn * vn
    vn3 = vn2 * vn
    
    # Time terms
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Construct the equation
    # Drive
    a = p0 * sin_t + p1 * cos_t
    
    # Stiffness (Duffing)
    a = a + p2 * xn + p3 * xn3
    
    # Damping (Linear + Cubic saturation)
    a = a + p4 * vn + p5 * vn3
    
    # Nonlinear Damping (Van der Pol) and Coupling
    a = a + p6 * xn2 * vn
    a = a + p7 * xn * vn
    
    # Quadratic position and offset
    a = a + p8 * xn2
    a = a + p9
    
    return a
# EVOLVE-BLOCK-END
