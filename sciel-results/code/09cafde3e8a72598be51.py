"""Symbolic regression: E. coli growth rate with Monod + temp/pH response."""
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 150.0  # safety margin before 150s kill


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate: Monod substrate saturation × temp response × pH response × density effect."""
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Ensure numerical safety: avoid division by zero, overflow
    # Monod substrate term: S / (Ks + S), Ks = exp(p1) to keep positive
    Ks = np.exp(np.clip(p1, -10, 10))
    monod = s / (Ks + s)

    # Temperature response: Gaussian around optimal temp
    # temp_opt = p2, width = exp(p3)
    temp_opt = p2
    temp_width = np.exp(np.clip(p3, -5, 5))
    temp_resp = np.exp(-0.5 * ((temp - temp_opt) / temp_width) ** 2)

    # pH response: Gaussian around optimal pH
    ph_opt = p4
    ph_width = np.exp(np.clip(p5, -5, 5))
    ph_resp = np.exp(-0.5 * ((pH - ph_opt) / ph_width) ** 2)

    # Maximum growth rate
    mu_max = np.exp(np.clip(p0, -10, 10))

    # Density effect: logistic inhibition
    # At low b, growth is near mu_max; at high b, inhibited
    # K_carry = exp(p6)
    K_carry = np.exp(np.clip(p6, -10, 10))
    density_factor = 1.0 - (b / (K_carry + b))

    # Combine: mu = mu_max * monod * temp_resp * ph_resp * density_factor
    mu = mu_max * monod * temp_resp * ph_resp * density_factor

    # Add small linear correction terms using remaining params
    # p7, p8, p9 for fine-tuning
    correction = p7 * (s / (Ks + s + 1)) + p8 * (temp - temp_opt) / (temp_width + 1) + p9 * (pH - ph_opt) / (ph_width + 1)

    result = mu + correction

    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    return result
