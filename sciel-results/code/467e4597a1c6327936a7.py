import numpy as np
import time

_START_TIME = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # x is expected to be in [0.1, 6]
    # Linear-in-params model with a rich, well-conditioned basis:
    #   p[0]: constant
    #   p[1]: x
    #   p[2]: x^2
    #   p[3]: x^3
    #   p[4]: x^4
    #   p[5]: sin(x)
    #   p[6]: cos(x)
    #   p[7]: sin(2x)
    #   p[8]: cos(2x)
    #   p[9]: exp(x/3)
    #
    # Because the model is linear in params, BFGS (or any convex optimizer)
    # converges to the global optimum of this fixed basis in very few steps,
    # avoiding the rugged landscape of nonlinear RBF width optimization.
    #
    # The basis spans:
    #   - Polynomials up to degree 4 (smooth trends, curvature)
    #   - Two frequencies of sine/cosine (periodic structure)
    #   - One broad exponential (saturation / growth)
    
    x = np.asarray(x, dtype=np.float64)
    
    # Polynomial terms
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    # Trigonometric terms
    sin1 = np.sin(x)
    cos1 = np.cos(x)
    sin2 = np.sin(2.0 * x)
    cos2 = np.cos(2.0 * x)
    
    # Exponential term (broad, slow growth)
    exp_term = np.exp(x / 3.0)
    
    y = (params[0]
         + params[1] * x
         + params[2] * x2
         + params[3] * x3
         + params[4] * x4
         + params[5] * sin1
         + params[6] * cos1
         + params[7] * sin2
         + params[8] * cos2
         + params[9] * exp_term)
    
    return y
