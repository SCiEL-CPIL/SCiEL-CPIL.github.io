"""Stress-strain of an aluminium rod: closed-form elastic-plastic + hardening model."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    stress = E(T)*e + H(T)*(e - eps_y)^(n(T)) + K(T)*exp(-e/alpha) + c0 + c1*T + c2*T^2

    with e = strain + eps0 (small positive offset for finiteness),
    E(T) = p0 + p1*T,  H(T) = p2 + p3*T,  n(T) = p4 + p5*T,  K(T) = p6 + p7*T.

    All 10 params are used; the form is smooth and closed-form for BFGS.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, c0, c1 = params

    eps0 = 1.0e-6
    e = strain + eps0                      # shift for finiteness / differentiability
    e = np.clip(e, 1.0e-9, None)          # keep power base strictly positive

    # Temperature-dependent coefficients
    E = p0 + p1 * temp                     # elastic modulus (thermal softening)
    H = p2 + p3 * temp                     # hardening magnitude
    n = p4 + p5 * temp                     # hardening exponent
    K = p6 + p7 * temp                     # transient hardening magnitude

    # Elastic contribution
    elastic = E * e

    # Ramberg-Osgood plastic hardening (power law)
    plastic = H * np.power(e, n)

    # Exponential transient hardening (sharp post-yield rise)
    transient = K * np.exp(-e / 1.0e-2)

    # Temperature baseline terms (constant + linear)
    base = c0 + c1 * temp

    stress = elastic + plastic + transient + base
    return stress
# EVOLVE-BLOCK-END
