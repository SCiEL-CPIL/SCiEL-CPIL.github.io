"""Symbolic regression with composite nonlinear parametric model."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear parametric model with 10 parameters:
    - Polynomial trend: c0, c1, c2, c3 (degree 3)
    - Gaussian bump: amp_g, center_g, width_g
    - Sine wave: amp_s, freq_s, phase_s
    
    y = c0 + c1*x + c2*x^2 + c3*x^3 + amp_g * exp(-0.5 * ((x - center_g) / width_g)^2) + amp_s * sin(freq_s * x + phase_s)
    
    Parameters are constrained to be safe:
    - width_g is transformed via softplus to ensure positivity.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Unpack parameters
    c0, c1, c2, c3 = params[0], params[1], params[2], params[3]
    amp_g, center_g, width_g_raw = params[4], params[5], params[6]
    amp_s, freq_s, phase_s = params[7], params[8], params[9]
    
    # Ensure width is positive and stable
    # softplus: log(1 + exp(x))
    # Clip raw value to avoid overflow in exp
    width_g_raw_clipped = np.clip(width_g_raw, -50, 50)
    width_g = np.log1p(np.exp(width_g_raw_clipped)) + 1e-6
    
    # Polynomial part
    poly = c0 + c1 * x + c2 * x**2 + c3 * x**3
    
    # Gaussian part
    # Avoid division by zero issues, width_g is >= 1e-6
    gauss_arg = -0.5 * ((x - center_g) / width_g) ** 2
    # Clip argument to avoid underflow issues, though exp handles it
    gauss_arg = np.clip(gauss_arg, -700, 700)
    gauss = amp_g * np.exp(gauss_arg)
    
    # Sine part
    sine = amp_s * np.sin(freq_s * x + phase_s)
    
    # Combine
    y = poly + gauss + sine
    
    # Ensure finite output
    y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return y
# EVOLVE-BLOCK-END
