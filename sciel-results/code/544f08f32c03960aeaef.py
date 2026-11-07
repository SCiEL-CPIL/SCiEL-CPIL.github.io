import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a hybrid basis combining polynomial/rational
    global structure with local Gaussian RBF kernels for non-polynomial features.
    
    Parameter allocation (10 total):
      p[0]: constant term
      p[1]: x coefficient
      p[2]: y coefficient
      p[3]: z coefficient
      p[4]: x*y coefficient
      p[5]: x*z coefficient
      p[6]: y*z coefficient
      p[7]: weight of Gaussian kernel 1
      p[8]: weight of Gaussian kernel 2
      p[9]: shared width (sigma) for both Gaussian kernels
      
    The first 7 parameters form a global polynomial basis (up to degree 2, no x^2, y^2, z^2
    to save parameters for the RBF part). The last 3 parameters control two Gaussian kernels
    with centers fixed at strategic points in the domain [0.5, 2]^3.
    """
    p = np.asarray(params, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    # Global polynomial basis terms
    t0 = np.ones_like(x)
    t1 = x
    t2 = y
    t3 = z
    t4 = x * y
    t5 = x * z
    t6 = y * z
    
    # Gaussian RBF kernels with fixed centers
    # Center 1: near lower-left corner of domain
    c1x, c1y, c1z = 0.7, 0.7, 0.7
    # Center 2: near upper-right corner of domain
    c2x, c2y, c2z = 1.8, 1.8, 1.8
    
    # Shared width, ensure positive
    sigma = np.maximum(np.abs(p[9]), 0.1)
    
    # Compute squared distances
    d1 = (x - c1x)**2 + (y - c1y)**2 + (z - c1z)**2
    d2 = (x - c2x)**2 + (y - c2y)**2 + (z - c2z)**2
    
    # Gaussian kernels
    g1 = np.exp(-d1 / (2.0 * sigma**2))
    g2 = np.exp(-d2 / (2.0 * sigma**2))
    
    # Linear combination
    result = (p[0] * t0 +
              p[1] * t1 +
              p[2] * t2 +
              p[3] * t3 +
              p[4] * t4 +
              p[5] * t5 +
              p[6] * t6 +
              p[7] * g1 +
              p[8] * g2)
    
    return result
