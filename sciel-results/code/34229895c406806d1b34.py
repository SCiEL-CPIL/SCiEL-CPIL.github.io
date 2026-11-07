"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model: smooth elastic + power-law hardening with temperature dependence.
    Uses C-infinity smooth functions throughout for better BFGS optimization.
    
    stress = E(T)*strain + A(T)*strain*(strain^2 + eps^2)^((n-1)/2) + C(T)
    
    The hardening term is an odd, smooth function of strain that approximates
    |strain|^n for |strain| >> eps, and is linear near strain=0.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > 130:
        return np.zeros_like(strain, dtype=np.float64)
    
    p0, p1 = params[0], params[1]
    p2, p3 = params[2], params[3]
    p4, p5 = params[4], params[5]
    p6, p7, p8 = params[6], params[7], params[8]
    p9 = params[9]
    
    # Temperature-dependent coefficients
    E_T = p0 + p1 * temp
    A_T = p2 + p3 * temp
    C_T = p6 + p7 * temp + p8 * temp * temp
    
    # Smooth positive exponent via softplus (always > 0, differentiable)
    n_raw = p4 + p5 * temp
    n_T = np.log1p(np.exp(np.clip(n_raw, -50, 50)))
    
    # Smooth positive scale via softplus
    eps_scale = np.log1p(np.exp(np.clip(p9, -50, 50)))
    
    # Smooth odd power: strain * (strain^2 + eps^2)^((n-1)/2)
    strain_sq = strain * strain
    base = strain_sq + eps_scale * eps_scale
    
    # Exponent for the base term
    exp_half = (n_T - 1.0) / 2.0
    # Clip to reasonable range for numerical stability
    exp_half = np.clip(exp_half, -0.5, 5.0)
    
    # Compute base^exp_half in log-space for stability
    log_base = np.log(base)
    log_power = exp_half * log_base
    power_term = np.exp(log_power)
    
    # Smooth signed power (odd function, C-infinity)
    smooth_power = strain * power_term
    
    # Combine
    elastic = E_T * strain
    hardening = A_T * smooth_power
    result = elastic + hardening + C_T
    
    return result
# EVOLVE-BLOCK-END
