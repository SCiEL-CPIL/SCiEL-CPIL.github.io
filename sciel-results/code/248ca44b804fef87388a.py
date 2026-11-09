# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 125.0  # safety margin before 150s kill

# Precomputed RBF centers on a 2D grid of (x, v)
# x range: [-0.2056, 0.1749], v range: [-0.3391, 0.3624]
_RBF_CENTERS_X = np.array([-0.12, 0.0, 0.12])
_RBF_CENTERS_V = np.array([-0.22, 0.0, 0.22])
_RBF_SIGMA = 0.25


def _compute_rbf(x, v, cx, cv, sigma):
    """Compute Gaussian RBF value at (x, v) centered at (cx, cv)."""
    dx = x - cx
    dv = v - cv
    return np.exp(-(dx * dx + dv * dv) / (2.0 * sigma * sigma))


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid RBF + sinusoidal model for nonlinear oscillator acceleration.
    
    Structure:
    - 4 sinusoidal drive terms: sin(t), cos(t), sin(t)*x, cos(t)*v
    - 6 RBF-local terms: 3 Gaussian bumps each multiplied by (1 + x + v)
      to capture local nonlinear coupling in the phase plane.
    
    The RBF centers are placed on a 2D grid covering the training range.
    Each RBF term is of the form: RBF(x,v) * (1 + x + v)
    This allows local linear approximation of the acceleration field.
    
    Total: 10 parameters (4 for drive, 6 for RBF-local).
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    n = len(t)
    
    # Sinusoidal drive terms
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # RBF-local terms: each is RBF(x,v) * (1 + x + v)
    # This captures local nonlinear coupling
    
    rbf0 = _compute_rbf(x, v, _RBF_CENTERS_X[0], _RBF_CENTERS_V[0], _RBF_SIGMA)
    rbf1 = _compute_rbf(x, v, _RBF_CENTERS_X[1], _RBF_CENTERS_V[1], _RBF_SIGMA)
    rbf2 = _compute_rbf(x, v, _RBF_CENTERS_X[2], _RBF_CENTERS_V[2], _RBF_SIGMA)
    rbf3 = _compute_rbf(x, v, _RBF_CENTERS_X[0], _RBF_CENTERS_V[2], _RBF_SIGMA)
    rbf4 = _compute_rbf(x, v, _RBF_CENTERS_X[2], _RBF_CENTERS_V[0], _RBF_SIGMA)
    rbf5 = _compute_rbf(x, v, _RBF_CENTERS_X[1], _RBF_CENTERS_V[0], _RBF_SIGMA)
    
    # Local linear factor
    local_linear = 1.0 + x + v
    
    # Build prediction
    # Params 0-3: sinusoidal drive
    # Params 4-9: RBF-local terms
    
    pred = (params[0] * sin_t +
            params[1] * cos_t +
            params[2] * sin_t * x +
            params[3] * cos_t * v +
            params[4] * rbf0 * local_linear +
            params[5] * rbf1 * local_linear +
            params[6] * rbf2 * local_linear +
            params[7] * rbf3 * local_linear +
            params[8] * rbf4 * local_linear +
            params[9] * rbf5 * local_linear)
    
    return pred
# EVOLVE-BLOCK-END
