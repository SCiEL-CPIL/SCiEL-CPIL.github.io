"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for an aluminium rod using a piecewise-linear
    elastic-plastic model with temperature-dependent yield and hardening.
    
    The model uses:
    - Linear elastic region up to yield
    - Linear hardening beyond yield
    - Temperature-dependent yield stress and modulus
    - Smooth transition at yield point
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure positive values for physical quantities
    E0 = np.abs(p0) + 0.1          # Base elastic modulus
    K_hard = np.abs(p1) + 0.01     # Hardening modulus
    sigma_y0 = np.abs(p2) + 0.01   # Base yield stress
    temp_coeff = p3                # Temperature coefficient for yield
    temp_E_coeff = p4              # Temperature coefficient for modulus
    eps_offset = p5                # Strain offset
    temp_offset = p6               # Temperature offset
    smooth_param = np.abs(p7) + 0.1  # Smoothing parameter
    quadratic_coeff = p8           # Quadratic strain correction
    linear_temp_coeff = p9         # Linear temperature effect on stress
    
    # Normalize temperature around 0.5 (midpoint of [0.06667, 1])
    t_norm = temp - 0.5
    
    # Temperature-dependent elastic modulus
    E_eff = E0 * (1.0 + temp_E_coeff * t_norm)
    
    # Temperature-dependent yield stress
    sigma_y = sigma_y0 * (1.0 + temp_coeff * t_norm)
    
    # Effective strain with offset
    eps_eff = strain + eps_offset
    
    # Strain at yield
    eps_y = sigma_y / E_eff
    
    # For each point, determine if in elastic or plastic region
    # Use smooth transition to avoid non-differentiability
    
    # Elastic stress
    sigma_elastic = E_eff * eps_eff
    
    # Plastic stress: yield + hardening * (strain - strain_at_yield)
    # For compression (negative strain), use symmetric behavior
    eps_sign = np.sign(eps_eff)
    eps_abs = np.abs(eps_eff)
    eps_y_abs = np.abs(eps_y)
    
    # Smooth min/max using softplus-like functions
    # plastic_strain = max(0, |eps| - |eps_y|)
    # Use softplus approximation: log(1 + exp(x)) for smooth max
    
    x = (eps_abs - eps_y_abs) / smooth_param
    # Softplus: log(1 + exp(x)), clipped for numerical stability
    x_clipped = np.clip(x, -50, 50)
    softplus_x = np.log1p(np.exp(x_clipped))
    plastic_strain = smooth_param * softplus_x
    
    # Plastic stress contribution
    sigma_plastic = sigma_y * eps_sign + K_hard * plastic_strain * eps_sign
    
    # Combine elastic and plastic using smooth switch
    # When |eps| < |eps_y|, use elastic; otherwise use plastic
    # Smooth switch: sigmoid based on (|eps| - |eps_y|)
    switch_arg = (eps_abs - eps_y_abs) / smooth_param
    switch_arg_clipped = np.clip(switch_arg, -50, 50)
    switch = 1.0 / (1.0 + np.exp(-switch_arg_clipped))
    
    # Interpolate between elastic and plastic
    sigma = (1.0 - switch) * sigma_elastic + switch * sigma_plastic
    
    # Add quadratic strain correction
    sigma += quadratic_coeff * strain ** 2
    
    # Add linear temperature effect
    sigma += linear_temp_coeff * t_norm
    
    return sigma
# EVOLVE-BLOCK-END
