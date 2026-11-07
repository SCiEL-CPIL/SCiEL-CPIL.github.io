"""Symbolic regression: stress-strain of an aluminium rod."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model using Voce-type exponential hardening.
    
    sigma = E(T)*eps + sigma_y(T) + Q(T)*(1 - exp(-eps/H(T)))
    
    Where:
    - E(T) = p0 + p1*T + p2*T^2  (elastic modulus)
    - sigma_y(T) = p3 + p4*T     (yield stress)
    - Q(T) = p5 + p6*T           (saturation stress)
    - H(T) = p7 + p8*T           (hardening modulus / characteristic strain)
    - p9 is an offset
    
    This is linear in parameters, so BFGS converges quickly.
    """
    s = strain
    t = temp
    p = params
    
    # Elastic part
    E_T = p[0] + p[1] * t + p[2] * t * t
    elastic = E_T * s
    
    # Yield stress
    sigma_y = p[3] + p[4] * t
    
    # Saturation stress
    Q_T = p[5] + p[6] * t
    
    # Characteristic strain (hardening modulus)
    # Ensure H is positive to avoid numerical issues
    H_T = p[7] + p[8] * t
    # Clip to avoid zero or negative values in exponential
    H_safe = np.where(np.abs(H_T) > 1e-10, H_T, np.sign(H_T + 1e-10) * 1e-10)
    
    # Exponential hardening term
    # For large positive eps/H, exp(-eps/H) -> 0, so term -> Q
    # For negative eps, we need to be careful
    # Use a safe version: clip the argument to avoid overflow
    arg = -s / H_safe
    arg_clipped = np.clip(arg, -50, 50)
    exp_term = np.exp(arg_clipped)
    
    hardening = Q_T * (1.0 - exp_term)
    
    # Offset
    offset = p[9]
    
    stress = elastic + sigma_y + hardening + offset
    
    return stress
# EVOLVE-BLOCK-END
