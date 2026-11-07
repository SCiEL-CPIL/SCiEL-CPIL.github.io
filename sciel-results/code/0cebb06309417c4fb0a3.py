"""Piecewise-linear stress-strain model with a smooth hinge for the elastic/plastic transition."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.

    Uses a smooth sigmoid-based hinge to separate elastic and plastic regimes.
    Elastic contribution: E * s * (1 - sigmoid(k*(s - s0)))
    Plastic contribution: Y * sigmoid(k*(s - s0)) * (1 - exp(-h*(s - s0)))
    Temperature modulates both E and Y.
    """
    s = np.asarray(strain, dtype=float)
    t = np.asarray(temp, dtype=float)

    # Unpack the 10 fitted constants
    E = params[0]          # Elastic modulus (initial slope)
    k = params[1]          # Hinge sharpness (transition width)
    Y = params[2]          # Yield stress (plateau level)
    s0 = params[3]         # Yield strain location
    H = params[4]          # Post-yield hardening rate
    c1 = params[5]         # Temp coefficient for Y softening
    c2 = params[6]         # Temp coefficient for E softening
    t0 = params[7]         # Reference temperature
    off = params[8]        # Constant offset
    k_t = params[9]        # Temp modulation of hinge sharpness

    # Temperature factors
    dt = t - t0
    soft_Y = 1.0 + c1 * dt
    soft_E = 1.0 + c2 * dt

    # Clamp to avoid negative moduli
    soft_Y = np.maximum(soft_Y, 0.05)
    soft_E = np.maximum(soft_E, 0.05)

    # Temperature-modulated hinge sharpness
    k_eff = k * (1.0 + k_t * dt)
    k_eff = np.maximum(np.abs(k_eff), 1e-4)

    # Smooth sigmoid hinge: approx 0 for s < s0, approx 1 for s > s0
    z = np.clip(k_eff * (s - s0), -50.0, 50.0)
    w = 1.0 / (1.0 + np.exp(-z))  # plastic weight

    # Elastic strain fraction: (1 - w)
    # Plastic strain fraction: w

    # Elastic stress
    stress_elastic = E * soft_E * s * (1.0 - w)

    # Plastic stress: yield plateau with hardening
    # For s > s0, plastic strain is (s - s0)
    s_plastic = np.maximum(s - s0, 0.0)
    # Hardening: linear after yield, capped by saturation
    L_safe = np.maximum(abs(H), 1e-6)
    hardening_term = H * s_plastic * (1.0 - np.exp(-L_safe * s_plastic))
    stress_plastic = Y * soft_Y * w + hardening_term * w

    stress = stress_elastic + stress_plastic + off

    return stress
# EVOLVE-BLOCK-END
