"""Tanh-Saturated Limit-Cycle Oscillator Model:
Replaces unbounded polynomial nonlinearities with bounded tanh functions to enforce
physical stability and prevent BFGS divergence. The structure follows a limit-cycle
oscillator form with position-velocity coupling and dual-frequency periodic forcing.
This constrains the solution space to stable dynamical regimes, improving optimization
robustness and potentially capturing nonlinear oscillator behavior more accurately."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 135.0
_GLOBAL = {}


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Tanh-saturated limit-cycle oscillator with 10 parameters:
    p0: offset
    p1: linear stiffness (x)
    p2: nonlinear stiffness saturation (tanh(x))
    p3: linear damping (v)
    p4: nonlinear damping saturation (tanh(v))
    p5: position-velocity coupling saturation (tanh(x*v))
    p6: sinusoidal forcing freq 0.5 rad/s (sin)
    p7: sinusoidal forcing freq 0.5 rad/s (cos)
    p8: sinusoidal forcing freq 1.0 rad/s (sin)
    p9: sinusoidal forcing freq 1.0 rad/s (cos)
    
    This model is linear in parameters. It covers:
    - Constant, linear, and saturated nonlinear spatial/velocity terms.
    - Saturated position-velocity interaction.
    - Two distinct time frequencies with arbitrary phase.
    - Bounded nonlinearities prevent unbounded growth and improve BFGS convergence.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Nonlinear spring and damping terms with tanh saturation
    spring = p1 * x + p2 * np.tanh(x)
    damping = p3 * v + p4 * np.tanh(v)
    coupling = p5 * np.tanh(x * v)
    
    # Multi-frequency periodic terms
    # Freq 0.5 rad/s (Period ~ 12.57)
    forcing_low = p6 * np.sin(0.5 * t) + p7 * np.cos(0.5 * t)
    
    # Freq 1.0 rad/s (Period ~ 6.28)
    forcing_high = p8 * np.sin(1.0 * t) + p9 * np.cos(1.0 * t)
    
    pred = p0 + spring + damping + coupling + forcing_low + forcing_high
    
    return pred
# EVOLVE-BLOCK-END
