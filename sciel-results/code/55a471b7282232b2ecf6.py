# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Local Linearization with Gaussian RBFs in the (x, v) phase plane.
    
    Structural change: instead of global polynomial/Fourier basis, use
    3 fixed Gaussian centers in (x, v) space. Each center provides a
    local linear model: w0 + w1*x + w2*v, weighted by the Gaussian.
    
    Parameters (10 total):
    - params[0:3]:  weights for center 1 (bias, x-coeff, v-coeff)
    - params[3:6]:  weights for center 2 (bias, x-coeff, v-coeff)
    - params[6:9]:  weights for center 3 (bias, x-coeff, v-coeff)
    - params[9]:    global bias term
    
    Fixed centers chosen to cover the training range:
    - Center 1: (x=-0.15, v=-0.15)  -> lower-left quadrant
    - Center 2: (x=0.0,  v=0.0)     -> origin
    - Center 3: (x=0.15, v=0.15)    -> upper-right quadrant
    
    Gaussian width sigma=0.25 to provide moderate locality.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    n = len(t)
    
    # Fixed Gaussian centers in (x, v) phase plane
    cx1, cv1 = -0.15, -0.15
    cx2, cv2 = 0.0, 0.0
    cx3, cv3 = 0.15, 0.15
    sigma = 0.25
    sigma2 = 2.0 * sigma * sigma
    
    # Compute Gaussian weights
    dx1 = x - cx1
    dv1 = v - cv1
    g1 = np.exp(-(dx1 * dx1 + dv1 * dv1) / sigma2)
    
    dx2 = x - cx2
    dv2 = v - cv2
    g2 = np.exp(-(dx2 * dx2 + dv2 * dv2) / sigma2)
    
    dx3 = x - cx3
    dv3 = v - cv3
    g3 = np.exp(-(dx3 * dx3 + dv3 * dv3) / sigma2)
    
    # Local linear models weighted by Gaussians
    # Center 1: w0 + w1*x + w2*v, scaled by g1
    local1 = g1 * (params[0] + params[1] * x + params[2] * v)
    
    # Center 2: w0 + w1*x + w2*v, scaled by g2
    local2 = g2 * (params[3] + params[4] * x + params[5] * v)
    
    # Center 3: w0 + w1*x + w2*v, scaled by g3
    local3 = g3 * (params[6] + params[7] * x + params[8] * v)
    
    # Global bias
    bias = params[9]
    
    pred = local1 + local2 + local3 + bias
    
    return pred
# EVOLVE-BLOCK-END
