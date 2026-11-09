"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _safe_exp(x, clip_val=500.0):
    """Numerically stable exp with clipping to avoid overflow."""
    return np.exp(np.clip(x, -clip_val, clip_val))


def _rbf_kernel(x, centers, widths):
    """
    Compute RBF kernel values.
    x: (n, 1) or (n,)
    centers: (k,)
    widths: (k,)
    Returns: (n, k)
    """
    x = np.asarray(x).reshape(-1, 1)
    centers = np.asarray(centers).reshape(-1, 1)
    widths = np.asarray(widths).reshape(-1, 1)
    
    # (x - c)^2 / (2*w^2)
    diff = x - centers  # (n, k)
    # Avoid division by zero
    w2 = np.maximum(widths**2, 1e-12)
    dist2 = diff**2 / (2.0 * w2)  # (n, k)
    
    return _safe_exp(-dist2)


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a hybrid model:
    - Polynomial basis (up to degree 4)
    - RBF basis with 3 centers
    - Trig basis (sin, cos)
    
    Params layout (length 10):
    0: poly coeff x^0 (intercept)
    1: poly coeff x^1
    2: poly coeff x^2
    3: poly coeff x^3
    4: poly coeff x^4
    5: RBF center 1
    6: RBF center 2
    7: RBF center 3
    8: RBF weight 1
    9: RBF weight 2
    (Note: We only have 10 params, so we use 2 RBF weights and fix the 3rd or use a different mapping)
    
    Revised mapping for 10 params:
    0: poly x^0
    1: poly x^1
    2: poly x^2
    3: poly x^3
    4: poly x^4
    5: rbf_center_1
    6: rbf_center_2
    7: rbf_weight_1
    8: rbf_weight_2
    9: rbf_width (shared width for simplicity, or use log-width to ensure positive)
    
    Let's try:
    0-4: Polynomial coefficients (degree 0-4)
    5: center_1
    6: center_2
    7: weight_1
    8: weight_2
    9: log_width (width = exp(log_width))
    
    This gives a flexible model.
    """
    x = np.asarray(x).reshape(-1)
    
    # Polynomial part
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    
    poly = p0 + p1 * x + p2 * x**2 + p3 * x**3 + p4 * x**4
    
    # RBF part
    c1 = params[5]
    c2 = params[6]
    w1 = params[7]
    w2 = params[8]
    log_w = params[9]
    
    # Ensure width is positive and reasonable
    width = np.exp(np.clip(log_w, -10, 10))
    
    # RBF kernels
    k1 = _safe_exp(-((x - c1) / width) ** 2 / 2.0)
    k2 = _safe_exp(-((x - c2) / width) ** 2 / 2.0)
    
    rbf = w1 * k1 + w2 * k2
    
    return poly + rbf


# EVOLVE-BLOCK-END
