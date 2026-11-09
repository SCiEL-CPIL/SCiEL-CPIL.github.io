"""Hybrid Physical-Polynomial and Dual-Frequency Fourier Model:
Retains core nonlinear oscillator terms (cubic stiffness, cubic damping,
coupling) and the fundamental frequency (1.0 rad/s) from the previous best
strategy, while adding a second lower frequency (0.5 rad/s) to capture
slower oscillatory modes or parametric variations. Drops less standard
time-interaction terms (t*x, t*v) to accommodate the richer time basis,
maintaining linearity in parameters for robust BFGS convergence."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 135.0
_GLOBAL = {}


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model with 10 parameters:
    p0: offset
    p1: linear stiffness (x)
    p2: cubic stiffness (x^3)
    p3: linear damping (v)
    p4: cubic damping (v^3)
    p5: position-velocity coupling (x*v)
    p6: sinusoidal forcing freq 0.5 rad/s (sin)
    p7: sinusoidal forcing freq 0.5 rad/s (cos)
    p8: sinusoidal forcing freq 1.0 rad/s (sin)
    p9: sinusoidal forcing freq 1.0 rad/s (cos)
    
    This model is linear in parameters. It covers:
    - Constant, linear, and cubic spatial/velocity nonlinearities.
    - Position-velocity interaction.
    - Two distinct time frequencies with arbitrary phase.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Nonlinear spring and damping terms
    spring = p1 * x + p2 * x**3
    damping = p3 * v + p4 * v**3
    coupling = p5 * x * v
    
    # Multi-frequency periodic terms
    # Freq 0.5 rad/s (Period ~ 12.57)
    forcing_low = p6 * np.sin(0.5 * t) + p7 * np.cos(0.5 * t)
    
    # Freq 1.0 rad/s (Period ~ 6.28)
    forcing_high = p8 * np.sin(1.0 * t) + p9 * np.cos(1.0 * t)
    
    pred = p0 + spring + damping + coupling + forcing_low + forcing_high
    
    return pred
# EVOLVE-BLOCK-END
