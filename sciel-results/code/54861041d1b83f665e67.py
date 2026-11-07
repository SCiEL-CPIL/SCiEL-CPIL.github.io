"""
Alternative strategy: Universal smooth functional form.
Instead of enforcing specific physical parameter roles (E(T), sigma_y(T)),
we use a truncated power series in strain and temperature.
This allows the optimizer to find the best-fitting algebraic structure
(linear, quadratic, etc.) directly from the data, avoiding potential
mis-specification of the physical model (e.g., if E(T) is not linear).
The form is symmetric in strain to handle compression/tension smoothly.
"""
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress prediction using a smooth, symmetric power series in strain and temp.
    
    The form is:
    sigma = a0 + a1*strain + a2*strain^2 + a3*strain^3 
          + b0*temp + b1*temp*strain + b2*temp*strain^2
          + c0*strain^4 + c1*temp^2*strain + c2*temp^2*strain^2
    
    This provides 10 parameters to fit.
    - a1 acts as the effective Young's Modulus (slope at 0).
    - a2, a3, c0 capture non-linear hardening/softening.
    - b terms capture temperature dependence of stiffness and non-linearity.
    - c1, c2 capture higher-order thermal effects.
    
    Using even powers of strain (strain^2, strain^4) ensures the function is 
    smooth and differentiable everywhere, avoiding sign issues with negative strain 
    that can plague fractional power laws (like Ramberg-Osgood).
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Pre-compute powers of strain and temp for efficiency
    s = strain
    t = temp
    
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    
    t2 = t * t
    
    # Construct the series
    # Elastic/Linear term
    term1 = p0 * s
    
    # Non-linear strain terms (symmetric hardening)
    term2 = p1 * s2
    term3 = p2 * s3
    term4 = p3 * s4
    
    # Temperature linear terms
    term5 = p4 * t
    term6 = p5 * t * s
    
    # Interaction terms (temp-dependent stiffness and hardening)
    term7 = p6 * t * s2
    
    # Higher order thermal effects
    term8 = p7 * t2 * s
    term9 = p8 * t2 * s2
    
    # Constant offset
    term10 = p9
    
    stress = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10
    
    return stress
