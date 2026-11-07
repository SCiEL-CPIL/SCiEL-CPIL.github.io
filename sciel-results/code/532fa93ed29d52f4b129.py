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
    
    # Elastic response
    elastic = E * strain
    
    # Plastic response with smooth hinge
    # Use smooth approximation: (strain - p2) * (1 - exp(-p4 * (strain - p2)))
    # This is 0 when strain <= p2 and grows smoothly for strain > p2
    plastic_strain = strain - p2
    # Ensure smoothness: use softplus-like function for the hinge
    # f(x) = x * (1 - exp(-k*x)) is not quite right for negative x
    # Better: use (x + softplus(-x)) where softplus(x) = log(1+exp(x))
    # But let's use a simpler smooth hinge: max(0, x) approximated
    # Using: x * sigmoid(k*x) which is 0 for x<<0 and ~x for x>>0
    k = p4
    # sigmoid(k*x) = 1 / (1 + exp(-k*x))
    # Handle large values to avoid overflow
    kx = k * plastic_strain
    # Clip to avoid overflow in exp
    kx_clipped = np.clip(kx, -500, 500)
    sigmoid = 1.0 / (1.0 + np.exp(-kx_clipped))
    smooth_plastic = plastic_strain * sigmoid
    
    plastic = p3 * smooth_plastic
    
    # Temperature-strain interaction terms
    temp_interaction = p5 * temp * strain + p6 * temp * temp * strain
    
    # Pure temperature terms
    temp_only = p7 * temp + p8 * temp * temp + p9
    
    stress = elastic + plastic + temp_interaction + temp_only
    
    return stress
# EVOLVE-BLOCK-END
