"""Piecewise-linear stress-strain model with a smooth hinge for the elastic/plastic transition."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.

    Uses a smooth hinge based on tanh to separate elastic and plastic regimes.
    Elastic part: E * (s - h) where h is the plastic strain component.
    Plastic part: Y + H * h which adds yield stress and linear hardening.
    Temperature softens Y, E, and modulates hinge sharpness.
    """
    s = np.asarray(strain, dtype=float)
    t = np.asarray(temp, dtype=float)

    # Unpack the 10 fitted constants
    E = params[0]          # Elastic modulus
    k = params[1]          # Hinge sharpness
    Y = params[2]          # Yield stress offset
    s_y = params[3]        # Yield strain
    H = params[4]          # Post-yield hardening slope
    c1 = params[5]         # Temp softening for Yield
    c2 = params[6]         # Temp softening for Modulus
    t0 = params[7]         # Reference temperature
    off = params[8]        # Offset
    k_t = params[9]        # Temp modulation of hinge sharpness

    # Temperature factors
    dt = t - t0
    soft_Y = 1.0 - c1 * dt
    soft_E = 1.0 - c2 * dt
    
    # Ensure softening factors don't go negative or cause issues
    soft_Y = np.maximum(soft_Y, 0.1)
    soft_E = np.maximum(soft_E, 0.1)

    # Modulate k with temperature
    k_eff = k * (1.0 + k_t * dt)
    k_eff = np.maximum(k_eff, 1e-6)

    # Smooth hinge: approximates max(0, s - s_y)
    x = s - s_y
    z = k_eff * x
    # Clip z to avoid overflow in tanh, though tanh is safe, it's good practice
    z = np.clip(z, -50.0, 50.0)
    h = x * (0.5 + 0.5 * np.tanh(z))
    
    # h is the plastic strain component
    # For s < s_y, h ~ 0. For s > s_y, h ~ s - s_y.
    
    # Elastic strain is total minus plastic
    e_elastic = s - h

    # Yield stress scaled by temp
    Y_eff = Y * soft_Y
    
    # Stress components
    stress = E * soft_E * e_elastic + Y_eff + H * h + off

    return stress
# EVOLVE-BLOCK-END
