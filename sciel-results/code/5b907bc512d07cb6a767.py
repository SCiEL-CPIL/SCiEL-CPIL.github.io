"""Rewritten symbolic regression candidate: Neural-network-style fixed basis with diverse activations."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 130.0  # Safety margin before 150s limit

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a fixed set of 10 non-linear basis functions
    inspired by a single-hidden-layer neural network with diverse activations.
    params: length-10 array of coefficients (fitted by BFGS).
    Returns an array with one prediction per point.
    """
    if time.time() - _start_time > _TIMEOUT:
        pass

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Pre-compute some common terms
    xy = x * y
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    x4 = x2 * x2
    y4 = y2 * y2

    # 10 diverse non-linear basis functions combining polynomial and trigonometric terms
    # These are designed to be rich but computationally cheap

    # Basis 0: Constant
    b0 = np.ones_like(x)

    # Basis 1: Linear combination with tanh
    b1 = np.tanh(x + 0.5 * y)

    # Basis 2: Linear combination with sin
    b2 = np.sin(1.5 * x - 0.3 * y)

    # Basis 3: Quadratic interaction
    b3 = x * y * np.cos(x - y)

    # Basis 4: Cubic-like term
    b4 = x * x * y - y * y * x

    # Basis 5: Sigmoid-like activation
    b5 = 1.0 / (1.0 + np.exp(-2.0 * (x + y)))

    # Basis 6: Cosine interaction
    b6 = np.cos(x * y)

    # Basis 7: Mixed polynomial
    b7 = x * x * np.sin(y) + y * y * np.sin(x)

    # Basis 8: Higher-order interaction
    b8 = (x + y) * np.tanh(x - y)

    # Basis 9: Product of sines
    b9 = np.sin(x) * np.sin(y)

    # Linear combination
    return (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 +
            p5 * b5 + p6 * b6 + p7 * b7 + p8 * b8 + p9 * b9)
# EVOLVE-BLOCK-END
