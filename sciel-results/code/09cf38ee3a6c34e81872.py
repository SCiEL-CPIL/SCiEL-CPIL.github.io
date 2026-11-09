import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Neural ODE-style model for nonlinear oscillator acceleration.
    
    Structure: a = v * tanh(p0) - x * (p1 + p2 * tanh(p3 * x^2 + p4 * v)) + p5 * v * tanh(p6 * v) + p7 * x * v * tanh(p8 * x) + p9 * x * v^2
    
    This directly encodes state-dependent damping and stiffness nonlinearity
    characteristic of driven Duffing/van der Pol oscillators, avoiding
    ill-conditioning from high-order polynomial expansions.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to [-1, 1] based on training data ranges
    x_mid = -0.02265
    x_half = 0.55015
    v_mid = 0.05335
    v_half = 0.22875
    
    xn = (x - x_mid) / x_half
    vn = (v - v_mid) / v_half
    
    # State-dependent stiffness: p1 + p2 * tanh(p3 * x^2 + p4 * v)
    x2 = xn * xn
    stiffness_mod = p3 * x2 + p4 * vn
    stiffness = p1 + p2 * np.tanh(stiffness_mod)
    
    # Damping terms
    # Base linear damping with saturation
    damp_base = p0 * np.tanh(p0)
    # Nonlinear velocity-dependent damping
    damp_nl = p5 * np.tanh(p6 * vn)
    # Cross coupling term
    cross = p7 * np.tanh(p8 * xn)
    
    # Construct the model
    # Damping: v * (base + nonlinear)
    damping = vn * (damp_base + damp_nl)
    
    # Stiffness: -x * state_dependent_stiffness
    stiff_term = -xn * stiffness
    
    # Cross-coupling: x * v * modulated
    cross_term = xn * vn * cross
    
    # Higher-order modulation
    mod_term = p9 * xn * vn * vn
    
    return stiff_term + damping + cross_term + mod_term
