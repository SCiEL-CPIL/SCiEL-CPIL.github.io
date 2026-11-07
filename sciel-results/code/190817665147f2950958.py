"""Piecewise-linear growth plateau with rational inhibition for E. coli growth rate."""
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model.

    Model:
      mu = p0 * temp_factor * pH_factor * substrate_factor * density_factor + p9

    Parameters (length 10):
      p[0]: max growth rate multiplier
      p[1]: T_min
      p[2]: T_opt
      p[3]: T_max
      p[4]: pH_min
      p[5]: pH_opt
      p[6]: pH_max
      p[7]: Ks (substrate half-saturation, linear term)
      p[8]: Ki (density inhibition scale)
      p[9]: offset
    """
    p = params
    eps = 1e-6

    # --- Temperature factor: triangular (tent) shape via clip product ---
    T_min = p[1]
    T_opt = p[2]
    T_max = p[3]

    width_up_T = np.maximum(T_opt - T_min, eps)
    width_down_T = np.maximum(T_max - T_opt, eps)

    rise_T = np.clip((temp - T_min) / width_up_T, 0.0, 1.0)
    fall_T = np.clip((T_max - temp) / width_down_T, 0.0, 1.0)
    temp_factor = rise_T * fall_T

    # --- pH factor: triangular (tent) shape via clip product ---
    pH_min = p[4]
    pH_opt = p[5]
    pH_max = p[6]

    width_up_ph = np.maximum(pH_opt - pH_min, eps)
    width_down_ph = np.maximum(pH_max - pH_opt, eps)

    rise_ph = np.clip((pH - pH_min) / width_up_ph, 0.0, 1.0)
    fall_ph = np.clip((pH_max - pH) / width_down_ph, 0.0, 1.0)
    pH_factor = rise_ph * fall_ph

    # --- Substrate factor: combined linear + quadratic saturation ---
    Ks = np.abs(p[7]) + eps
    # Linear Monod term
    mono = s / (Ks + s)
    # Quadratic saturation term (reuses Ks as scale for the quadratic)
    quad = (s * s) / (Ks * Ks + s * s)
    # Blend: mostly linear, small quadratic contribution for shape flexibility
    substrate_factor = 0.8 * mono + 0.2 * quad

    # --- Density inhibition: tanh-based smooth saturation ---
    Ki = np.abs(p[8]) + eps
    density_factor = 1.0 - np.tanh(b / Ki)

    # --- Combine ---
    mu = p[0] * temp_factor * pH_factor * substrate_factor * density_factor + p[9]

    return mu
