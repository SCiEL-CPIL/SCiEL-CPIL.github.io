"""Symbolic regression for E. coli growth rate - GAM with B-spline-like basis."""
import numpy as np


def _legendre_basis(x):
    """
    Compute 3 Legendre polynomial basis functions scaled to [0, 1].
    Returns array of shape (..., 3).
    """
    # Shift to [-1, 1]
    t = 2.0 * x - 1.0
    # Legendre polynomials: P0=1, P1=t, P2=(3t^2-1)/2, P3=(5t^3-3t)/2
    # Use P1, P2, P3 (skip P0 since it's constant and captured by bias)
    p1 = t
    p2 = 0.5 * (3.0 * t ** 2 - 1.0)
    p3 = 0.5 * (5.0 * t ** 3 - 3.0 * t)
    return np.stack([p1, p2, p3], axis=-1)


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Generalized additive model (GAM) for E. coli growth rate with 10 parameters.
    Uses Monod kinetics for substrate, and Legendre polynomial basis functions
    for temperature and pH to capture non-monotonic responses.
    
    Parameters:
        p[0]: mu_max - maximum growth rate scale
        p[1]: log_Ks - log of Monod half-saturation constant
        p[2:5]: coefficients for temperature basis (3 Legendre polys)
        p[5:8]: coefficients for pH basis (3 Legendre polys)
        p[8]: biomass linear coefficient
        p[9]: bias
    """
    p = params
    mu_max = p[0]
    log_Ks = p[1]
    c_t = p[2:5]   # 3 coefficients for temperature
    c_p = p[5:8]   # 3 coefficients for pH
    c_b = p[8]
    bias = p[9]
    
    # Normalize inputs to [0, 1] range
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 39.0
    temp_norm = (temp - 1.0) / 11.0
    ph_norm = (pH - 1.0) / 4.0
    
    # Clip normalized inputs to [0, 1] for stability
    b_norm = np.clip(b_norm, 0.0, 1.0)
    s_norm = np.clip(s_norm, 0.0, 1.0)
    temp_norm = np.clip(temp_norm, 0.0, 1.0)
    ph_norm = np.clip(ph_norm, 0.0, 1.0)
    
    # Ensure positive Ks via log-parameterization
    Ks = np.exp(log_Ks) + 1e-8
    
    # Monod kinetics for substrate saturation
    monod_s = s_norm / (Ks + s_norm + 1e-12)
    
    # Compute Legendre basis for temperature and pH
    temp_basis = _legendre_basis(temp_norm)  # shape: (..., 3)
    ph_basis = _legendre_basis(ph_norm)      # shape: (..., 3)
    
    # Additive temperature and pH contributions
    temp_add = np.sum(c_t * temp_basis, axis=-1)  # shape: (...)
    ph_add = np.sum(c_p * ph_basis, axis=-1)      # shape: (...)
    
    # Biomass linear effect
    b_term = 1.0 + c_b * b_norm
    
    # Combine: multiplicative structure with additive corrections
    # mu = mu_max * monod(s) * (1 + temp_add) * (1 + ph_add) * b_term + bias
    temp_factor = 1.0 + temp_add
    ph_factor = 1.0 + ph_add
    
    # Ensure factors don't go too negative (soft clamp)
    temp_factor = np.where(temp_factor > 0.01, temp_factor, 0.01)
    ph_factor = np.where(ph_factor > 0.01, ph_factor, 0.01)
    
    mu = mu_max * monod_s * temp_factor * ph_factor * b_term + bias
    
    # Clip to avoid extreme values
    mu = np.clip(mu, -10.0, 10.0)
    
    return mu
