"""Stress-strain model for aluminium rod with temperature dependence."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form stress model for aluminium rod.
    
    Uses a smooth bilinear model with temperature-dependent modulus and
    yield point, plus higher-order corrections.
    
    params (length 10):
        p0: base elastic modulus
        p1: temperature coefficient of modulus (per unit temp)
        p2: yield stress at reference temp
        p3: temperature coefficient of yield stress
        p4: plastic hardening modulus
        p5: plastic hardening temp coefficient
        p6: strain at yield (offset)
        p7: quadratic strain correction
        p8: interaction term (strain * temp)
        p9: intercept
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent elastic modulus
    E = p0 + p1 * temp
    
    # Temperature-dependent yield stress
    sigma_y = p2 + p3 * temp
    
    # Smooth approximation of max(0, strain - strain_yield) using softplus-like
    # Actually, use a smooth hinge: for strain > strain_yield, plastic region
    # We'll use: plastic_strain = max(0, strain - p6) approximated smoothly
    # Use: 0.5 * (strain - p6 + sqrt((strain - p6)^2 + eps)) - 0.5*sqrt(eps^2)
    # But eps is fixed; let's use a simpler smooth max via tanh or just piecewise
    # For BFGS, we need smoothness. Use:
    # softplus(x) = log(1 + exp(x)) / k for large k
    # But that can overflow. Alternative: use (x + sqrt(x^2 + delta))/2 with small delta
    
    delta = 1e-6
    x = strain - p6
    plastic_strain = 0.5 * (x + np.sqrt(x * x + delta) - np.sqrt(delta))
    
    # Temperature-dependent plastic modulus
    H = p4 + p5 * temp
    
    # Stress = elastic part + plastic part + corrections
    # Elastic part: E * strain (but capped at yield, then plastic takes over)
    # Smooth bilinear: sigma = E * min(strain, strain_y) + H * max(0, strain - strain_y)
    # In terms of our parameters, strain_y is implicit via p6
    # Let's use: sigma = E * strain - E * plastic_strain + H * plastic_strain
    #          = E * strain + (H - E) * plastic_strain
    
    sigma = E * strain + (H - E) * plastic_strain
    
    # Add quadratic strain correction
    sigma += p7 * strain * strain
    
    # Add interaction term
    sigma += p8 * strain * temp
    
    # Add intercept
    sigma += p9
    
    return sigma
# EVOLVE-BLOCK-END
