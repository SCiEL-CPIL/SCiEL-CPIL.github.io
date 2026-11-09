"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a physics-motivated multiplicative model:
      - Elastic region: sigma = E * eps
      - Plastic region: sigma = K * eps^n * (1 - alpha*(T - T0))
      - Smooth transition via tanh
      - Additional correction terms for non-ideal behavior
    """
    _t0 = time.perf_counter()
    _DEADLINE = 140.0

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)

    # Clamp strain to avoid issues with negative values in power law
    # Physical strain for plastic part should be positive
    eps_plastic = np.maximum(strain, 1e-12)

    # Elastic stress
    sigma_elastic = p0 * strain

    # Temperature correction factor (multiplicative)
    # p1: temperature sensitivity, p2: reference temperature offset
    temp_corr = 1.0 + p1 * (temp - p2)
    # Ensure temp_corr stays positive for numerical stability
    temp_corr = np.maximum(temp_corr, 0.01)

    # Plastic stress: Hollomon-type power law
    # sigma_plastic = K * eps^n * temp_corr
    # p3: K (strength coefficient), p4: n (hardening exponent)
    # Use exp(n*log(eps)) for numerical stability
    log_eps = np.log(eps_plastic)
    sigma_plastic = p3 * np.exp(p4 * log_eps) * temp_corr

    # Smooth transition between elastic and plastic
    # p5: transition sharpness, p6: yield strain threshold
    transition = 0.5 * (1.0 + np.tanh(p5 * (strain - p6)))

    # Combined stress with smooth blend
    sigma = (1.0 - transition) * sigma_elastic + transition * sigma_plastic

    # Correction terms for non-ideal behavior
    # p7, p8, p9: additional correction factors
    # Quadratic strain correction
    corr1 = p7 * strain * strain
    # Temperature-strain interaction correction
    corr2 = p8 * strain * (temp - p2)
    # Higher-order temperature correction
    corr3 = p9 * (temp - p2) * (temp - p2) * strain

    prediction = sigma + corr1 + corr2 + corr3

    if time.perf_counter() - _t0 > _DEADLINE:
        pass

    return prediction
# EVOLVE-BLOCK-END
