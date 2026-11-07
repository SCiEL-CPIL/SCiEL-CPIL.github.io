"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-weights composite basis: polynomial + fixed-frequency trig.
    
    All 10 parameters are linear weights, making the fitting problem
    convex (least-squares). BFGS from all-ones converges to the global
    optimum reliably.
    
    Basis functions (fixed, no nonlinear params):
      p0: 1
      p1: x
      p2: x^2
      p3: x^3
      p4: x^4
      t0: sin(x)
      t1: cos(x)
      t2: sin(2x)
      t3: cos(2x)
      t4: sin(3x)
    
    prediction(x) = sum_i w_i * basis_i(x)
    """
    x = np.asarray(x, dtype=np.float64)
    p = params  # shape (10,)
    
    # Polynomial terms
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    # Trig terms (fixed frequencies 1, 2, 3)
    s1 = np.sin(x)
    c1 = np.cos(x)
    s2 = np.sin(2.0 * x)
    c2 = np.cos(2.0 * x)
    s3 = np.sin(3.0 * x)
    
    # Linear combination with learnable weights
    prediction = (
        p[0] * 1.0
        + p[1] * x
        + p[2] * x2
        + p[3] * x3
        + p[4] * x4
        + p[5] * s1
        + p[6] * c1
        + p[7] * s2
        + p[8] * c2
        + p[9] * s3
    )
    
    return prediction
# EVOLVE-BLOCK-END
