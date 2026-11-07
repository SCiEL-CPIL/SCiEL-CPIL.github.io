"""Piecewise-linear stress-strain model with a smooth hinge for the elastic/plastic transition."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.

    Uses a smooth hinge h = s * (1 - exp(-k*s)) to separate elastic and plastic regimes.
    Elastic part: E * (s - h) which decays as plasticity sets in.
    Plastic part: Y * (1 - exp(-h/L)) which saturates at yield stress Y.
    Hardening: H * (h - L) * (1 - exp(-h/L)) adds post-yield strength.
    Temperature softens Y and E.
    """
    s = np.asarray(strain, dtype=float)
    t = np.asarray(temp, dtype=float)

    # Unpack the 10 fitted constants
    E = params[0]          # Elastic modulus
    k = params[1]          # Hinge sharpness
    Y = params[2]          # Yield stress
    L = params[3]          # Characteristic plastic strain scale
    H = params[4]          # Post-yield hardening coefficient
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

    # Smooth hinge
    # h approx 0 for small s, h approx s for large s
    z = np.clip(k_eff * s, -50.0, 50.0)
    h = s * (1.0 - np.exp(-z))

    # Elastic strain is the part not "plasticized"
    e_elastic = s - h

    # Plastic stress components
    # Saturation term
    L_safe = np.maximum(L, 1e-6)
    z_plastic = np.clip(h / L_safe, -50.0, 50.0)
    sat = 1.0 - np.exp(-z_plastic)
    
    # Yield stress scaled by temp
    Y_eff = Y * soft_Y
    
    # Hardening term: activates as h exceeds L
    hardening = H * (h - L_safe) * sat

    stress = E * soft_E * e_elastic + Y_eff * sat + hardening + off

    return stress
# EVOLVE-BLOCK-END
