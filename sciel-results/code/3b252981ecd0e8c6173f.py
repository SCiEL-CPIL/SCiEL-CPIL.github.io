"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a Voce-type exponential hardening model combined with a smooth
    elastic-plastic transition.

    Model:
        sigma = sigma_y(T) + Q(T) * (1 - exp(-eps_plastic / H(T)))
    
    where:
        sigma_y(T) = p0 + p1 * T + p2 * T^2   (yield stress as quadratic in temp)
        Q(T)       = p3 + p4 * T + p5 * T^2   (saturation stress as quadratic in temp)
        H(T)       = p6 + p7 * T              (hardening modulus, linear in temp)
    
    The plastic strain is computed as max(0, strain - eps_y(T)) where eps_y(T) is
    the yield strain. To keep things smooth and differentiable for BFGS, we use
    a softplus-like transition:
        eps_plastic = strain - eps_y(T)
        smooth_plastic = 0.5 * (eps_plastic + sqrt(eps_plastic^2 + p8^2))
    
    This gives a smooth transition from elastic to plastic behavior.
    p8 controls the sharpness of the transition (small p8 = sharper).
    p9 is an additional offset.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Ensure parameters are used in a numerically stable way
    # Yield strain: we model it as a small constant scaled by p9
    # Actually, let's define yield strain as a parameter derived from the model
    # For simplicity, let's say the elastic region ends at a small strain
    # We can use p9 as a scaling for the yield strain
    # Let's define: eps_y = p9 * 0.001  (small strain, around 0.001 scale)
    # But p9 starts at 1.0, so eps_y = 0.001 initially
    
    # Temperature-dependent yield strain
    eps_y = p9 * 0.001
    
    # Plastic strain (smooth)
    eps_plastic = strain - eps_y
    # Softplus for smoothness: 0.5 * (x + sqrt(x^2 + delta^2))
    # This is smooth and differentiable everywhere
    delta = 0.0001  # small regularization for numerical stability
    smooth_eps_plastic = 0.5 * (eps_plastic + np.sqrt(eps_plastic**2 + delta**2))
    
    # Ensure non-negative (the softplus already gives non-negative for x >= 0)
    # but for x < 0, it gives a small positive value, which is fine
    
    # Temperature-dependent parameters
    sigma_y = p0 + p1 * temp + p2 * temp**2
    Q = p3 + p4 * temp + p5 * temp**2
    H = p6 + p7 * temp
    
    # Avoid division by zero or negative H
    # Clip H to a small positive value
    H_safe = np.where(np.abs(H) < 1e-10, 1e-10, H)
    
    # Voce hardening term
    # 1 - exp(-eps_plastic / H)
    # For numerical stability, clip the exponent
    exponent = -smooth_eps_plastic / H_safe
    # Clip to avoid overflow
    exponent = np.clip(exponent, -50, 50)
    hardening = 1.0 - np.exp(exponent)
    
    # Total stress
    stress = sigma_y + Q * hardening
    
    return stress
# EVOLVE-BLOCK-END
