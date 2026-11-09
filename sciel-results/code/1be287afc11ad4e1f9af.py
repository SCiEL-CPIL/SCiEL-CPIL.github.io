"""Aluminium rod stress-strain model with a soft saturation (sigmoidal) yield transition."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    A physically-motivated model for aluminium stress-strain with temperature dependence.
    The model captures:
      - An initial elastic region (linear in strain)
      - A smooth yield transition
      - A post-yield hardening region that saturates
    We use a smooth "kink" via a sigmoid to blend elastic and plastic behaviour,
    with all coefficients temperature-dependent via linear functions of temp.

    Model:
      stress = E(T)*strain * S(T, strain) + Y(T)*tanh(K(T)*strain + b(T))
    where
      S(T, strain) = 1 / (1 + c(T) * strain^2)  -- softens the elastic slope at large strain
    E(T)  = e0 + e1*temp
      Y(T)  = y0 + y1*temp
      K(T)  = k0 + k1*temp
      c(T)  = c0 + c1*temp
      b(T)  = b0 + b1*temp

    This uses 10 parameters: e0,e1,y0,y1,k0,k1,c0,c1,b0,b1.

    The tanh term provides the yield point and hardening plateau.
    The rational softening term prevents over-prediction at large strains.
    """
    e0, e1, y0, y1, k0, k1, c0, c1, b0, b1 = params

    # Temperature-dependent coefficients
    E = e0 + e1 * temp
    Y = y0 + y1 * temp
    K = k0 + k1 * temp
    c = c0 + c1 * temp
    b = b0 + b1 * temp

    # Softening factor (reduces elastic slope at large strain)
    # Use a gentle rational form
    soft = 1.0 / (1.0 + np.abs(c) * strain**2)

    # Sigmoidal yield contribution (smooth kink)
    # tanh saturates to ±1, giving a plateau after the yield point
    yield_term = Y * np.tanh(K * strain + b)

    # Elastic contribution with softening
    elastic = E * strain * soft

    return elastic + yield_term
# EVOLVE-BLOCK-END
