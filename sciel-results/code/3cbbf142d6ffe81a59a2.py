"""Physics-informed radial basis function (RBF) model for a damped nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with driving force.

    Uses 5 Gaussian RBF terms in the (x, v) plane. The 10 fitted parameters
    are the centers and widths of the 5 Gaussians:
        params[2*i]   -> x-center of term i
        params[2*i+1] -> v-center of term i
    A small linear base (fixed weights) is added so the model has non-zero,
    smooth gradients even when the RBF terms are far from the data, which
    helps BFGS converge from the all-ones starting point.
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    # Fixed linear base: guarantees smooth, non-zero gradients from any init.
    out = 0.1 * x + 0.1 * v

    # 5 Gaussian RBF terms in the (x, v) plane.
    for i in range(5):
        cx = p[2 * i]
        cv = p[2 * i + 1]
        dx = x - cx
        dv = v - cv
        out = out + 0.1 * np.exp(-0.5 * (dx * dx + dv * dv))

    return out
# EVOLVE-BLOCK-END
