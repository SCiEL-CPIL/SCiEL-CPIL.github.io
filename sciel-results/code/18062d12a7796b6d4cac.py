import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Neural ODE style model for nonlinear oscillator acceleration.
    
    Encodes the physical structure of a driven Duffing oscillator with:
    - State-dependent stiffness: x * (p1 + p2 * tanh(p3 * x^2 + p4 * v))
    - State-dependent damping: v * (p5 + p6 * tanh(p7 * x + p8 * v))
    - Bias and residual terms
    
    This directly models acceleration as restoring_force + damping_force,
    where both forces have nonlinear state-dependent coefficients.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to [-1, 1] based on training data ranges
    x_mid = -0.02265
    x_half = 0.55015
    v_mid = 0.05335
    v_half = 0.22875
    
    xn = (x - x_mid) / x_half
    vn = (v - v_mid) / v_half
    
    # State-dependent stiffness coefficient
    # Base stiffness p1, modulated by tanh of (x^2 + v coupling)
    stiffness_mod = np.tanh(p3 * xn * xn + p4 * vn)
    stiffness = p1 + p2 * stiffness_mod
    
    # State-dependent damping coefficient
    # Base damping p5, modulated by tanh of (x + v coupling)
    damping_mod = np.tanh(p7 * xn + p8 * vn)
    damping = p5 + p6 * damping_mod
    
    # Restoring force (proportional to displacement)
    restoring = -xn * stiffness
    
    # Damping force (proportional to velocity)
    damping_force = -vn * damping
    
    # Bias and residual term
    bias = p0 + p9 * np.tanh(xn * vn)
    
    return restoring + damping_force + bias
