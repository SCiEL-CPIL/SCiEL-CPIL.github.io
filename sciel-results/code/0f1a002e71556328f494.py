import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(x)

    p = params

    # Scaling to [-1, 1] based on known training ranges
    # This ensures good conditioning for the BFGS optimizer
    x_min, x_max = -0.5728, 0.5275
    v_min, v_max = -0.1754, 0.2821

    x_range = x_max - x_min
    v_range = v_max - v_min

    if x_range < 1e-9:
        u = np.zeros_like(x)
    else:
        u = 2.0 * (x - x_min) / x_range - 1.0

    if v_range < 1e-9:
        w = np.zeros_like(v)
    else:
        w = 2.0 * (v - v_min) / v_range - 1.0

    # Physics-informed basis with bounded nonlinearities
    # 
    # For a nonlinear oscillator, the acceleration a = f(x, v) typically has:
    # 1. A restoring force term dependent on x (e.g., -k*x for linear, -sin(x) for pendulum)
    # 2. A damping term dependent on v (e.g., -c*v for linear, -gamma*v*|v| for quadratic)
    # 3. Possibly nonlinear coupling terms
    #
    # Using bounded nonlinearities like sin and tanh helps capture:
    # - Periodic restoring forces (sin)
    # - Saturation effects (tanh)
    # - Better conditioning than high-degree polynomials
    #
    # Basis functions (10 total):
    # 0: 1 (bias)
    # 1: sin(u) - primary restoring force (pendulum-like)
    # 2: u - linear restoring force
    # 3: u^3 - cubic nonlinearity in position (weak spring)
    # 4: tanh(u) - saturated restoring force
    # 5: w - linear damping
    # 6: sin(w) - nonlinear damping (oscillatory velocity dependence)
    # 7: tanh(w) - saturated damping
    # 8: u * w - position-velocity coupling
    # 9: u * sin(w) - nonlinear position-velocity coupling
    
    b0 = np.ones_like(u)
    b1 = np.sin(u)
    b2 = u
    b3 = u * u * u
    b4 = np.tanh(u)
    b5 = w
    b6 = np.sin(w)
    b7 = np.tanh(w)
    b8 = u * w
    b9 = u * np.sin(w)

    return (
        p[0] * b0 +
        p[1] * b1 +
        p[2] * b2 +
        p[3] * b3 +
        p[4] * b4 +
        p[5] * b5 +
        p[6] * b6 +
        p[7] * b7 +
        p[8] * b8 +
        p[9] * b9
    )
