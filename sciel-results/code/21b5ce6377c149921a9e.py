"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a driven nonlinear oscillator using a Scaled Duffing-Van der Pol model.
    
    Structural Change:
    Instead of using raw inputs which can lead to ill-conditioned basis functions 
    (e.g., x^3 is very small for x in [-0.3, 0.3]), we scale x and v to be O(1).
    This helps the BFGS optimizer converge by keeping the fitted parameters 
    in a reasonable magnitude range and improving the condition number of the 
    Hessian matrix during optimization.
    
    Model:
    a = p0*sin(t) + p1*cos(t)                          (Periodic drive)
      + p2*xn + p3*xn^3                                (Scaled Duffing stiffness)
      + p4*vn + p5*vn^3                                (Scaled damping)
      + p6*xn^2*vn                                     (Scaled Van der Pol term)
      + p7*xn*vn                                       (Scaled cross-coupling)
      + p8*xn^2                                        (Scaled quadratic position)
      + p9                                             (Constant offset)
    
    Where xn = x / 0.5 and vn = v / 0.5 to normalize the input ranges 
    (approx [-0.35, 0.35]) to roughly [-1, 1].
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to improve numerical conditioning for BFGS
    # The training range for x is [-0.3391, 0.3624] and v is [-0.3391, 0.3624]
    # Using 0.5 as the scale factor maps this to approx [-0.7, 0.7], which is O(1).
    xn = x * 2.0
    vn = v * 2.0
    
    # Precompute powers of normalized variables
    xn2 = xn * xn
    xn3 = xn2 * xn
    vn2 = vn * vn
    vn3 = vn2 * vn
    
    # Time terms: Keep raw t for sin/cos as the frequency is likely 1.0 rad/s or similar
    # and t is in [30, 50], so sin(t) is well-defined.
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Construct the equation using scaled basis functions
    a = p0 * sin_t + p1 * cos_t
    a = a + p2 * xn + p3 * xn3
    a = a + p4 * vn + p5 * vn3
    a = a + p6 * xn2 * vn
    a = a + p7 * xn * vn
    a = a + p8 * xn2
    a = a + p9
    
    return a
# EVOLVE-BLOCK-END
