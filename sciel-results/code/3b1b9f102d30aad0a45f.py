"""Multiplicative temperature-modulated elastic-plastic model for aluminium rod stress-strain."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model:
      E(T)   = p0 + p1*temp + p2*temp^2
      sigma_y(T) = p3 + p4*temp
      n(T)   = p5 + p6*temp
      stress = E(T)*strain + sigma_y(T)*tanh(n(T)*strain)
    
    The tanh term provides smooth saturation behavior for plastic hardening,
    while the multiplicative coupling of temperature-dependent coefficients
    with strain captures the physical interplay between temperature and
    elastic-plastic response.
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent elastic modulus (quadratic in temp)
    E = p0 + p1 * temp + p2 * temp * temp
    
    # Temperature-dependent yield stress (linear in temp)
    sigma_y = p3 + p4 * temp
    
    # Temperature-dependent hardening rate
    n = p5 + p6 * temp
    
    # Elastic contribution
    elastic = E * strain
    
    # Plastic hardening via smooth tanh saturation
    # Clip n*strain to avoid overflow in tanh for extreme values
    arg = n * strain
    arg = np.clip(arg, -50.0, 50.0)
    plastic = sigma_y * np.tanh(arg)
    
    # Additional correction terms using remaining params
    # p7, p8, p9 provide additional flexibility:
    # p7: additional linear strain term
    # p8: quadratic strain term
    # p9: temperature-strain interaction
    correction = p7 * strain + p8 * strain * strain + p9 * temp * strain
    
    prediction = elastic + plastic + correction
    
    return prediction
# EVOLVE-BLOCK-END
