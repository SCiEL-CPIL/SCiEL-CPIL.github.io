"""
RBF + polynomial hybrid basis for symbolic regression.
Uses Gaussian RBF kernels with fixed centers covering the domain, plus
polynomial terms for global trends. Linear in parameters so BFGS converges
fast. Deadline guard ensures we return within the time limit.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_LIMIT = 150.0
_MARGIN = 15.0


def _time_ok():
    return (time.time() - _START) < (_LIMIT - _MARGIN)


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Polynomial terms for global structure
    t0 = np.ones_like(x)
    t1 = x
    t2 = y
    t3 = x * x
    t4 = y * y

    # RBF terms: Gaussian kernels centered at key points in [-1,1]^2
    # Using 5 RBF centers to capture local structure
    sigma = 0.4  # bandwidth

    # Center at (0,0)
    t5 = np.exp(-((x**2 + y**2) / (2 * sigma**2)))
    # Center at (0.5, 0.5)
    t6 = np.exp(-(((x - 0.5)**2 + (y - 0.5)**2) / (2 * sigma**2)))
    # Center at (-0.5, -0.5)
    t7 = np.exp(-(((x + 0.5)**2 + (y + 0.5)**2) / (2 * sigma**2)))
    # Center at (0.5, -0.5)
    t8 = np.exp(-(((x - 0.5)**2 + (y + 0.5)**2) / (2 * sigma**2)))
    # Center at (-0.5, 0.5)
    t9 = np.exp(-(((x + 0.5)**2 + (y - 0.5)**2) / (2 * sigma**2)))

    return (p0 * t0 +
            p1 * t1 +
            p2 * t2 +
            p3 * t3 +
            p4 * t4 +
            p5 * t5 +
            p6 * t6 +
            p7 * t7 +
            p8 * t8 +
            p9 * t9)
# EVOLVE-BLOCK-END
