"""Stress-strain model for aluminium rod with temperature dependence."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model using smooth hinge to separate elastic and plastic regimes:
      E(T) = p0 + p1 * temp
      yield_strain = p2
      hardening_rate = p3
      plastic_strain = strain - yield_strain (for strain > yield_strain)
      stress = E(T) * strain
               + hardening_rate * (strain - yield_strain) * (1 - exp(-p4 * (strain - yield_strain)))
               + p5 * temp * strain
               + p6 * temp^2 * strain
               + p7 * temp
               + p8 * temp^2
               + p9
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = p
    
    # Temperature-dependent elastic modulus
    E = p0 + p1 * temp
    
    # Elastic component
    elastic = E * strain
    
    # Plastic component with smooth hinge
    plastic_strain = strain - p2
    # Use softplus-like smooth hinge: x * (1 - exp(-k*x)) for x > 0, ~0 for x <= 0
    # This is differentiable everywhere
    k = p4
    # Smooth approximation of max(0, x) * (1 - exp(-k*x))
    # For numerical stability, clip plastic_strain to avoid overflow in exp
    ps_clipped = np.clip(plastic_strain, -10.0, 10.0)
    hinge = ps_clipped * (1.0 - np.exp(-k * ps_clipped))
    plastic = p3 * hinge
    
    # Temperature-strain interaction terms
    temp_interaction = p5 * temp * strain + p6 * temp * temp * strain
    
    # Pure temperature terms
    temp_only = p7 * temp + p8 * temp * temp + p9
    
    stress = elastic + plastic + temp_interaction + temp_only
    
    return stress
# EVOLVE-BLOCK-END
