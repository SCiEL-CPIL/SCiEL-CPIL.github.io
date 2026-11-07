"""Refined symbolic regression model for nonlinear oscillator with time-varying coefficients."""
import numpy as np
import time


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator with time-varying stiffness and damping via Fourier series.
    
    Model: a = (c0 + c1*sin(w*t_c) + c2*cos(w*t_c)) * x
           + (c3 + c4*sin(w*t_c) + c5*cos(w*t_c)) * v
           + c6 * x^3
           + c7 * v^3
           + c8 * x * v
           + c9 / (1 + alpha * (x^2 + v^2))
    
    where t_c = t - 40.0, w is a fixed frequency for numerical stability,
    and alpha is derived from a parameter to keep 10 total params.
    
    Actually, let me restructure to use exactly 10 params:
    c0: base stiffness
    c1: stiffness sin amplitude
    c2: stiffness cos amplitude  
    c3: base damping
    c4: damping sin amplitude
    c5: damping cos amplitude
    c6: cubic stiffness (Duffing)
    c7: cubic damping (Van der Pol)
    c8: coupling x*v
    c9: rational saturation coefficient
    """
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params

    # Center time for numerical conditioning
    t_c = t - 40.0

    # Fixed frequency for Fourier terms (avoid fitting frequency to keep params for dynamics)
    # Use a moderate frequency that can capture variations over the [30,50] window
    w = 0.5  # radians per time unit, gives ~3 cycles over 20 time units

    sin_w = np.sin(w * t_c)
    cos_w = np.cos(w * t_c)

    # Time-varying stiffness coefficient
    k_t = c0 + c1 * sin_w + c2 * cos_w

    # Time-varying damping coefficient
    d_t = c3 + c4 * sin_w + c5 * cos_w

    # Core nonlinear terms
    term_x = k_t * x
    term_v = d_t * v
    term_x3 = c6 * x**3
    term_v3 = c7 * v**3
    term_xv = c8 * x * v

    # Rational saturation term to capture bounded effects
    # Use a smooth denominator to avoid division by zero issues
    r2 = x**2 + v**2
    # Clamp denominator to avoid numerical issues
    denom = 1.0 + c9 * r2
    # Ensure denominator stays positive by using abs or clamping
    denom_safe = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    term_sat = 1.0 / denom_safe

    result = term_x + term_v + term_x3 + term_v3 + term_xv + term_sat

    # Safety: return zeros if any non-finite values appear
    if not np.all(np.isfinite(result)):
        return np.zeros_like(result)

    return result
