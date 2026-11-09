"""Symbolic regression with 2 inputs: localized RBF network with Gaussian kernels on a fixed grid."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 145.0

# Fixed centers for RBF: 3x3 grid on [-1, 1]^2
# Centers: (-1,-1), (0,-1), (1,-1), (-1,0), (0,0), (1,0), (-1,1), (0,1), (1,1)
# That's 9 centers. We have 10 params.
# Let's use 9 RBF weights + 1 global bias.
# Or we could use a smaller grid. Let's stick with 3x3 = 9 centers + 1 bias = 10 params.

_CENTERS = np.array([
    [-1.0, -1.0],
    [ 0.0, -1.0],
    [ 1.0, -1.0],
    [-1.0,  0.0],
    [ 0.0,  0.0],
    [ 1.0,  0.0],
    [-1.0,  1.0],
    [ 0.0,  1.0],
    [ 1.0,  1.0]
])

# Gaussian width parameter. 
# If width is too small, RBFs are very local. If too large, they overlap heavily.
# For a 3x3 grid on [-1,1], distance between neighbors is 1.0.
# A width of ~0.5-1.0 seems reasonable. 
# Since we can't tune this with BFGS (it's a hyperparameter), we pick a fixed value.
# Let's try sigma = 0.7. The kernel is exp(-||x-c||^2 / (2*sigma^2)).
_SIGMA2 = 2.0 * (0.7 ** 2)  # 2 * sigma^2


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Localized Radial Basis Function (RBF) network with 10 parameters.
    
    params = [w1, w2, ..., w9, b]
    
    pred = sum_{i=1}^{9} w_i * exp(-||[x,y] - c_i||^2 / (2*sigma^2)) + b
    
    This model is linear in the parameters, so BFGS will converge quickly and robustly.
    The Gaussian kernels allow capturing local variations and non-smooth features 
    that global polynomials might struggle with, while avoiding the ill-conditioning
    of high-degree polynomials.
    """
    if time.time() > _DEADLINE:
        # Fallback: simple linear model using first 3 params
        return params[0] * x + params[1] * y + params[2]

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    n = x.shape[0]
    
    # Compute squared distances from each point to each center
    # x, y are (n,), _CENTERS is (9, 2)
    # dist_sq[i, j] = (x[i] - cx[j])^2 + (y[i] - cy[j])^2
    cx = _CENTERS[:, 0]
    cy = _CENTERS[:, 1]
    
    # Vectorized computation
    # dx = x[:, None] - cx[None, :]  -> (n, 9)
    # dy = y[:, None] - cy[None, :]  -> (n, 9)
    dx = x[:, None] - cx[None, :]
    dy = y[:, None] - cy[None, :]
    dist_sq = dx * dx + dy * dy
    
    # Compute Gaussian kernels
    # K[i, j] = exp(-dist_sq[i, j] / _SIGMA2)
    K = np.exp(-dist_sq / _SIGMA2)
    
    # Prediction: K @ weights + bias
    # K is (n, 9), weights are params[0:9], bias is params[9]
    pred = K @ params[:9] + params[9]
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        mask = np.isfinite(pred)
        pred[~mask] = 0.0

    return pred
# EVOLVE-BLOCK-END
