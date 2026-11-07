"""Physics-informed anisotropic RBF model for a damped nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator.

    Uses 2 anisotropic Gaussian RBF terms in the (x, v) plane with a linear base.
    The 10 fitted parameters are:
        params[0]    -> constant offset
        params[1]    -> linear x coefficient
        params[2]    -> linear v coefficient
        params[3]    -> x-center of RBF 1 (mapped via tanh)
        params[4]    -> v-center of RBF 1 (mapped via tanh)
        params[5]    -> amplitude of RBF 1
        params[6]    -> x-center of RBF 2 (mapped via tanh)
        params[7]    -> v-center of RBF 2 (mapped via tanh)
        params[8]    -> amplitude of RBF 2
        params[9]    -> shared width (mapped via exp)
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    # Linear base: captures global trend
    c = p[0]
    a = p[1]
    b = p[2]
    out = c + a * x + b * v

    # Shared width, mapped to positive values
    # p[9]=1 -> sigma = 0.1 * e^1 ~ 0.27
    sigma = 0.1 * np.exp(p[9])
    
    # RBF 1
    cx1 = 0.5 * np.tanh(p[3])
    cv1 = 0.3 * np.tanh(p[4])
    A1 = p[5]
    
    dx1 = (x - cx1) / sigma
    dv1 = (v - cv1) / sigma
    out = out + A1 * np.exp(-0.5 * (dx1 * dx1 + dv1 * dv1))

    # RBF 2
    cx2 = 0.5 * np.tanh(p[6])
    cv2 = 0.3 * np.tanh(p[7])
    A2 = p[8]
    
    dx2 = (x - cx2) / sigma
    dv2 = (v - cv2) / sigma
    out = out + A2 * np.exp(-0.5 * (dx2 * dx2 + dv2 * dv2))

    return out
# EVOLVE-BLOCK-END
