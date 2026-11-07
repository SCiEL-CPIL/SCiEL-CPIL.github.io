"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Fixed scaling constants based on training ranges
X_CENTER = -0.02265
X_SCALE = 0.55015
V_CENTER = 0.05335
V_SCALE = 0.22875

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using tanh/sin/cos activation basis.
    
    Structural change: replace pure polynomial with bounded nonlinear activations
    that better capture oscillator physics (saturation, periodicity, damping).
    """
    # Scale inputs to O(1)
    xs = (x - X_CENTER) / X_SCALE
    vs = (v - V_CENTER) / V_SCALE
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Bounded nonlinear features for oscillator dynamics:
    # tanh captures saturation (soft Duffing), sin/cos capture periodicity
    # Cross terms capture coupling between position and velocity
    
    f0 = np.ones_like(xs)
    f1 = np.tanh(xs)           # soft linear restoring
    f2 = np.tanh(vs)           # soft velocity damping
    f3 = np.tanh(xs**2)        # quadratic restoring (bounded)
    f4 = np.tanh(xs**3)        # cubic restoring (Duffing-like, bounded)
    f5 = np.sin(xs)            # periodic restoring
    f6 = np.sin(vs)            # periodic velocity effect
    f7 = np.tanh(xs * vs)      # cross coupling
    f8 = np.cos(xs) * vs       # velocity modulated by position phase
    f9 = np.tanh(xs**2 * vs)   # nonlinear cross term
    
    result = (
        p0 * f0 +
        p1 * f1 +
        p2 * f2 +
        p3 * f3 +
        p4 * f4 +
        p5 * f5 +
        p6 * f6 +
        p7 * f7 +
        p8 * f8 +
        p9 * f9
    )
    
    return result
# EVOLVE-BLOCK-END
