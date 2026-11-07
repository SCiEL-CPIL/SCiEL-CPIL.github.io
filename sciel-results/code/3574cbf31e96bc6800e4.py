import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Tanh-based smooth step stress-strain model for aluminium rod.

    The stress is modeled as:
        sigma = E*eps * (1 - w(t)*tanh(k(t)*eps)) + sigma_plateau(t) * tanh(k(t)*eps) + H(t)*eps*tanh(k(t)*eps)

    where:
      - E = p0 + p1*temp          (elastic modulus, temp-dependent)
      - k = p2 + p3*temp          (transition sharpness, temp-dependent)
      - w = p4 + p5*temp          (elastic fraction that is "lost" at large strain)
      - sigma_plateau = p6 + p7*temp  (plateau stress level)
      - H = p8 + p9*temp          (post-yield hardening slope)

    The tanh function provides a strictly smooth, bounded transition from
    elastic (tanh->0, stress ~ E*eps) to a plateau+hardening regime
    (tanh->1, stress ~ E*(1-w)*eps + sigma_plateau + H*eps).
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)

    # Temperature-dependent coefficients
    E = p0 + p1 * temp
    k = p2 + p3 * temp
    w = p4 + p5 * temp
    sigma_plat = p6 + p7 * temp
    H = p8 + p9 * temp

    # Smooth step: tanh(k * strain)
    # For large positive strain, tanh -> 1; for small strain, tanh ~ k*strain
    # Clamp k*strain to avoid overflow in tanh
    ks = k * strain
    # np.tanh is safe for all finite inputs, but let's be cautious
    step = np.tanh(ks)

    # Elastic term: E*strain * (1 - w*step)
    # At strain=0: E*0 = 0
    # At small strain: E*strain * (1 - w*k*strain) ~ E*strain (elastic)
    # At large strain: E*strain * (1 - w) (reduced elastic contribution)
    elastic_term = E * strain * (1.0 - w * step)

    # Plateau term: sigma_plat * step
    # At small strain: ~ 0
    # At large strain: sigma_plat
    plateau_term = sigma_plat * step

    # Hardening term: H * strain * step
    # At small strain: ~ 0
    # At large strain: H * strain (linear hardening)
    hardening_term = H * strain * step

    stress = elastic_term + plateau_term + hardening_term

    return stress
