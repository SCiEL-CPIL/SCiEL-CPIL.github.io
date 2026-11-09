"""Smooth non-linear stress-strain model for aluminium rod."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Model: Smooth non-linear combination of elastic, yield, and hardening terms.
    
    stress = E(T)*strain + sigma_y(T)*(1 - exp(-k(T)*strain)) + p6*strain^2 + p7*strain^3 + p8*strain^4 + p9*strain*temp
    
    Where:
    E(T) = p0 * (1 - p1 * temp)
    sigma_y(T) = p2 * (1 - p3 * temp)
    k(T) = p4 * (1 - p5 * temp)
    
    This model is fully differentiable, which is suitable for BFGS optimization.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent coefficients
    E_T = p0 * (1.0 - p1 * temp)
    sigma_y_T = p2 * (1.0 - p3 * temp)
    k_T = p4 * (1.0 - p5 * temp)
    
    # Elastic term
    elastic = E_T * strain
    
    # Yield/Plastic term with smooth transition
    # Clip the exponent to prevent overflow/underflow issues during optimization
    exponent = -k_T * strain
    # If exponent is very large positive, exp is huge. If very large negative, exp is 0.
    # np.exp handles this, but we can be safe.
    # However, if k_T * strain is very large positive, 1 - exp(...) becomes very negative.
    # This might be physically unrealistic but mathematically valid for the fitter to explore.
    # To ensure stability, we might clip the exponent argument to [-500, 500]
    exponent_clipped = np.clip(exponent, -500.0, 500.0)
    plastic = sigma_y_T * (1.0 - np.exp(exponent_clipped))
    
    # Hardening terms
    hardening = p6 * strain**2 + p7 * strain**3 + p8 * strain**4
    
    # Temperature interaction term
    temp_interact = p9 * strain * temp
    
    prediction = elastic + plastic + hardening + temp_interact
    
    return prediction
# EVOLVE-BLOCK-END
