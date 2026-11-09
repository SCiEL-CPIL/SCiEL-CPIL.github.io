"""Structured multiplicative nonlinear oscillator: compositional form where
nonlinear coupling arises from products of displacement and velocity sub-expressions,
giving BFGS a well-conditioned landscape with physically meaningful parameter roles."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Closed-form prediction for a nonlinear oscillator using a structured
    multiplicative form.

    Structure:
        displacement_term = p0 + p1*x + p2*x^2 + p3*x^3
        velocity_term     = p4 + p5*v + p6*v^2
        coupling          = displacement_term * velocity_term
        time_modulation   = p7 + p8*t + p9*t^2
        prediction        = coupling * time_modulation

    This captures:
    - Linear damping (p1 * p5 * t via coupling * time_mod)
    - Nonlinear stiffness (p3 * p4 * t)
    - Velocity-dependent restoring (p1 * p6 * t)
    - Cross-coupling x^2 * v, x * v^2, etc.
    - Temporal modulation of all interactions
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Displacement sub-expression: captures restoring force nonlinearity
    x_term = p0 + p1 * x + p2 * x * x + p3 * x * x * x

    # Velocity sub-expression: captures damping nonlinearity
    v_term = p4 + p5 * v + p6 * v * v

    # Multiplicative coupling: generates all cross terms x^i * v^j
    coupling = x_term * v_term

    # Time modulation: allows amplitude decay/growth and phase drift
    t_term = p7 + p8 * t + p9 * t * t

    # Final prediction
    pred = coupling * t_term

    return pred

# EVOLVE-BLOCK-END
