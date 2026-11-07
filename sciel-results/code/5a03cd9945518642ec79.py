"""Symbolic regression: nonlinear oscillator (2) - Hybrid Linear + RBF model."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator.
    
    Model:
    y = p0 + p1*x + p2*v + p3*t + sum_{i=0}^{5} p_{4+i} * RBF_i(t, x, v)
    
    RBF_i are Gaussian functions with fixed centers and anisotropic widths.
    Centers are chosen to cover the input domain.
    """
    # Unpack parameters
    p0, p1, p2, p3 = params[0], params[1], params[2], params[3]
    w0, w1, w2, w3, w4, w5 = params[4], params[5], params[6], params[7], params[8], params[9]

    # Linear part
    y = p0 + p1 * x + p2 * v + p3 * t

    # RBF part
    # Centers (t_c, x_c, v_c)
    # Domain: t [30, 50], x [-0.2, 0.17], v [-0.34, 0.36]
    
    # Center 0: Early time, negative x/v
    t_c, x_c, v_c = 30.0, -0.2, -0.3
    dist_sq = (t - t_c)**2 / 200.0 + (x - x_c)**2 / 0.08 + (v - v_c)**2 / 0.18
    y += w0 * np.exp(-dist_sq)

    # Center 1: Late time, positive x/v
    t_c, x_c, v_c = 50.0, 0.17, 0.36
    dist_sq = (t - t_c)**2 / 200.0 + (x - x_c)**2 / 0.08 + (v - v_c)**2 / 0.18
    y += w1 * np.exp(-dist_sq)

    # Center 2: Mid time, center
    t_c, x_c, v_c = 40.0, 0.0, 0.0
    dist_sq = (t - t_c)**2 / 200.0 + (x - x_c)**2 / 0.08 + (v - v_c)**2 / 0.18
    y += w2 * np.exp(-dist_sq)

    # Center 3: Mid time, high x/v
    t_c, x_c, v_c = 40.0, 0.1, 0.2
    dist_sq = (t - t_c)**2 / 200.0 + (x - x_c)**2 / 0.08 + (v - v_c)**2 / 0.18
    y += w3 * np.exp(-dist_sq)

    # Center 4: Mid time, low x/v
    t_c, x_c, v_c = 40.0, -0.1, -0.2
    dist_sq = (t - t_c)**2 / 200.0 + (x - x_c)**2 / 0.08 + (v - v_c)**2 / 0.18
    y += w4 * np.exp(-dist_sq)

    # Center 5: Mid time, high x, low v
    t_c, x_c, v_c = 40.0, 0.1, -0.2
    dist_sq = (t - t_c)**2 / 200.0 + (x - x_c)**2 / 0.08 + (v - v_c)**2 / 0.18
    y += w5 * np.exp(-dist_sq)

    return y
# EVOLVE-BLOCK-END
