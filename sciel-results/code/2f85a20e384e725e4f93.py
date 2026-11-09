"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF model with 5 Gaussian kernels.
    Centers: 4 corners of [-1,1]^2 and the origin.
    params[0:5] = weights for each kernel
    params[5:10] = log-widths (log-sigma) for each kernel
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    n = x.shape[0]
    
    # Centers: (-1,-1), (1,-1), (-1,1), (1,1), (0,0)
    cx = np.array([-1.0, 1.0, -1.0, 1.0, 0.0])
    cy = np.array([-1.0, -1.0, 1.0, 1.0, 0.0])
    
    weights = params[0:5]
    log_sigmas = params[5:10]
    
    # Clip log_sigmas to avoid extreme values
    log_sigmas = np.clip(log_sigmas, -4, 4)
    sigmas = np.exp(log_sigmas)
    
    pred = np.zeros(n)
    for i in range(5):
        dx = x - cx[i]
        dy = y - cy[i]
        r2 = dx**2 + dy**2
        sigma = sigmas[i]
        kernel = np.exp(-r2 / (2.0 * sigma**2))
        pred += weights[i] * kernel
    
    return pred
# EVOLVE-BLOCK-END
