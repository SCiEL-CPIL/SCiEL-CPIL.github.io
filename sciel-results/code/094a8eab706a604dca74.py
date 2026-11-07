import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid linear basis: polynomial + exponential + trigonometric.
    
    y(x) = p0 + p1*x + p2*x^2 + p3*x^3
         + e1*exp(-x) + e2*exp(x)
         + t1*sin(x) + t2*cos(x)
         + r1*x*sin(x) + r2*x*cos(x)
    
    All 10 parameters enter linearly, making the least-squares problem
    convex and BFGS convergence reliable.
    """
    x = np.asarray(x, dtype=float)
    
    x2 = x * x
    x3 = x2 * x
    
    exp_neg = np.exp(-x)
    exp_pos = np.exp(x)
    sinx = np.sin(x)
    cosx = np.cos(x)
    xsinx = x * sinx
    xcosx = x * cosx
    
    y = (params[0]
         + params[1] * x
         + params[2] * x2
         + params[3] * x3
         + params[4] * exp_neg
         + params[5] * exp_pos
         + params[6] * sinx
         + params[7] * cosx
         + params[8] * xsinx
         + params[9] * xcosx)
    
    return y
