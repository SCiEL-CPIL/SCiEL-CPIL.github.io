# EVOLVE-BLOCK-START
import numpy as np
import time


def _softplus(x, delta):
    """Numerically stable softplus: delta * log(1 + exp(x/delta))"""
    # For numerical stability, use the identity:
    # softplus(x, d) = d * log1p(exp(x/d)) for x/d > -30
    # softplus(x, d) = x + d*log(1 + exp(-x/d)) for x/d <= -30
    x = np.asarray(x, dtype=np.float64)
    d = float(delta)
    if d <= 1e-12:
        return np.maximum(x, 0.0)
    arg = x / d
    # Use stable formulation
    out = np.empty_like(x)
    mask = arg > 30.0
    out[mask] = x[mask]
    out[~mask] = d * np.log1p(np.exp(arg[~mask]))
    return out


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model: Smooth-hinge Ramberg-Osgood
    sigma = E(T)*strain + H(T)*sign(strain)*softplus(|strain| - eps_y(T), delta)^p
    
    params:
    [0]: E0 (elastic modulus at ref T)
    [1]: E1 (linear T coefficient for E)
    [2]: E2 (quadratic T coefficient for E)
    [3]: H0 (hardening at ref T)
    [4]: H1 (linear T coefficient for H)
    [5]: ey0 (yield strain at ref T)
    [6]: ey1 (linear T coefficient for yield strain)
    [7]: delta (smoothness of yield transition)
    [8]: p (hardening exponent)
    [9]: C0 (constant offset)
    """
    E0, E1, E2 = params[0], params[1], params[2]
    H0, H1 = params[3], params[4]
    ey0, ey1 = params[5], params[6]
    delta = params[7]
    p = params[8]
    C0 = params[9]
    
    # Ensure delta is positive for softplus
    delta_safe = max(abs(delta), 1e-6)
    
    # Temperature-dependent Young's modulus (quadratic in T)
    E = E0 + E1 * temp + E2 * temp * temp
    
    # Temperature-dependent hardening modulus
    H = H0 + H1 * temp
    
    # Temperature-dependent yield strain
    eps_y = ey0 + ey1 * temp
    
    # Absolute strain and sign
    abs_strain = np.abs(strain)
    sign_strain = np.sign(strain)
    
    # Smooth hinge: softplus(|strain| - eps_y, delta)
    # This is ~ 0 when |strain| < eps_y, ~ |strain| - eps_y when |strain| >> eps_y
    hinge = _softplus(abs_strain - eps_y, delta_safe)
    
    # Hardening term: H * sign * hinge^p
    # For p in (0,1), this gives sub-linear hardening (typical for metals)
    # For p > 1, super-linear
    # Clip hinge to avoid negative values from numerical issues
    hinge_safe = np.maximum(hinge, 0.0)
    hardening = H * sign_strain * np.power(hinge_safe, p)
    
    # Elastic term
    elastic = E * strain
    
    # Combine
    stress = elastic + hardening + C0
    
    return stress
# EVOLVE-BLOCK-END
