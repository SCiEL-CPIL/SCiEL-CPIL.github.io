"""Physics-informed anisotropic RBF model for a damped nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator.

    Uses 2 isotropic Gaussian RBF terms with independent widths in the (x, v) plane
    and a linear base. The constant offset is absorbed into the RBFs.
    The 10 fitted parameters are:
        params[0]    -> linear x coefficient
        params[1]    -> linear v coefficient
        params[2]    -> x-center of RBF 1 (mapped via tanh)
        params[3]    -> v-center of RBF 1 (mapped via tanh)
        params[4]    -> amplitude of RBF 1
        params[5]    -> width of RBF 1 (mapped via exp)
        params[6]    -> x-center of RBF 2 (mapped via tanh)
        params[7]    -> v-center of RBF 2 (mapped via tanh)
        params[8]    -> amplitude of RBF 2
        params[9]    -> width of RBF 2 (mapped via exp)
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    # Linear base: captures global trend
    a = p[0]
    b = p[1]
    out = a * x + b * v

    # RBF 1
    cx1 = 0.5 * np.tanh(p[2])
    cv1 = 0.3 * np.tanh(p[3])
    A1 = p[4]
    sigma1 = 0.1 * np.exp(p[5])
    
    dx1 = (x - cx1) / sigma1
    dv1 = (v - cv1) / sigma1
    out = out + A1 * np.exp(-0.5 * (dx1 * dx1 + dv1 * dv1))

    # RBF 2
    cx2 = 0.5 * np.tanh(p[6])
    cv2 = 0.3 * np.tanh(p[7])
    A2 = p[8]
    sigma2 = 0.1 * np.exp(p[9])
    
    dx2 = (x - cx2) / sigma2
    dv2 = (v - cv2) / sigma2
    out = out + A2 * np.exp(-0.5 * (dx2 * dx2 + dv2 * dv2))

    return out
# EVOLVE-BLOCK-END
