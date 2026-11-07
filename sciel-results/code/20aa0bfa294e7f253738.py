import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a smooth bilinear model to separate elastic and plastic regimes.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent parameters
    E_e = p0 + p1 * temp       # Elastic modulus
    eps_y = p2 + p3 * temp     # Yield strain
    E_p = p4 + p5 * temp       # Plastic modulus (hardening)
    delta = p6 + p7 * temp     # Transition width
    
    # Ensure delta is not zero to avoid division by zero in tanh argument scaling
    # tanh(x/delta) is safe if delta is small, but if delta is 0, it's a step.
    # BFGS might find delta near 0. We should guard against it.
    delta_safe = np.where(np.abs(delta) < 1e-6, 1e-6, delta)
    
    # Smooth step function H
    # H is 0 for strain << eps_y and 1 for strain >> eps_y
    arg = (strain - eps_y) / delta_safe
    H = 0.5 * (1.0 + np.tanh(arg))
    
    # Bilinear stress model
    # Elastic part: E_e * strain
    # Plastic part: E_p * strain + (E_e - E_p) * eps_y
    # Blend: (1-H) * Elastic + H * Plastic
    elastic_stress = E_e * strain
    plastic_stress = E_p * strain + (E_e - E_p) * eps_y
    
    stress = (1.0 - H) * elastic_stress + H * plastic_stress
    
    # Add offset and quadratic correction
    stress = stress + p8 + p9 * strain * strain
    
    return stress
