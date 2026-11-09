# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model using Gaussian RBF with local linear terms.
    
    Uses 3 Gaussian centers in (x, v) phase space, each multiplied by a local
    linear function (1 + x + v) to capture non-additive nonlinear coupling.
    
    Inputs:
    t: Time
    x: Position
    v: Velocity
    params: Array of 10 fitted constants
        p0: global constant offset
        p1-p3: RBF center 1 weights (for 1, x, v local linear terms)
        p4-p6: RBF center 2 weights
        p7-p9: RBF center 3 weights
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Define 3 Gaussian RBF centers in (x, v) phase space
    centers_x = np.array([-0.10, 0.0, 0.08])
    centers_v = np.array([-0.15, 0.0, 0.15])
    
    # Gaussian bandwidth
    sigma = 0.15
    inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)
    
    # Compute Gaussian RBF values for each center (1D arrays)
    g1 = np.exp(-((x - centers_x[0])**2 + (v - centers_v[0])**2) * inv_2sigma2)
    g2 = np.exp(-((x - centers_x[1])**2 + (v - centers_v[1])**2) * inv_2sigma2)
    g3 = np.exp(-((x - centers_x[2])**2 + (v - centers_v[2])**2) * inv_2sigma2)
    
    # Local linear basis: each RBF multiplied by (1, x, v)
    result = (p0 +
              p1 * g1 + p2 * g1 * x + p3 * g1 * v +
              p4 * g2 + p5 * g2 * x + p6 * g2 * v +
              p7 * g3 + p8 * g3 * x + p9 * g3 * v)
    
    return result
# EVOLVE-BLOCK-END
