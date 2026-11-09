import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Combines a smooth rational hardening term (from current program) with
    tanh-based plastic saturation and multiplicative temperature modulation
    (from inspirations).
    
    Model:
      sigma = E(T) * eps * (1 + p4*eps^p5) / (1 + p6*eps^p7 + p8*T^p9)
              + S(T) * tanh(K(T) * eps)
              + Q(T) * eps^2
              + Offset(T)
    
    Where:
      E(T)   = p0 + p1*T          (Elastic Modulus, temperature dependent)
      S(T)   = p2 + p3*T          (Saturation Stress for tanh term)
      K(T)   = 1.0                (Fixed rate for tanh, or could be derived)
      Q(T)   = p2 * 0.5           (Quadratic correction, linked to S)
      Offset(T) = 0               (No separate offset to save params)
      
    Wait, I need to use exactly 10 params. Let's restructure to be coherent.
    
    Revised Model:
      sigma = (p0 + p1*T) * eps * (1 + p2*eps^p3) / (1 + p4*eps^p5 + p6*T^p7)
              + (p8 + p9*T) * tanh(eps / C0)
              
    This uses:
      p0, p1: Elastic modulus coefficients
      p2, p3: Non-linear strain factor numerator
      p4, p5: Non-linear strain factor denominator
      p6, p7: Temperature factor in denominator
      p8, p9: Saturation stress coefficients for tanh term
    
    This combines the multiplicative structure of the directive with the
    smooth tanh saturation from inspiration 1 and 3.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Elastic modulus
    E_T = p0 + p1 * T
    
    # Non-linear strain factor: (1 + p2*eps^p3) / (1 + p4*eps^p5 + p6*T^p7)
    # Ensure eps^p3 and eps^p5 are well-defined. Since eps can be negative,
    # we use sign-preserving power: sign(x)*|x|^p
    # But for physical stress-strain, eps is usually positive in plastic region.
    # Training range: [-0.001911, 0.9636]
    # To be safe, we use np.power(np.abs(eps), p) * np.sign(eps) for odd-like behavior,
    # or just clip eps to non-negative for the hardening part.
    # Let's use a safe power function.
    
    # For the denominator, we want it to be positive.
    # Let's simplify: use eps_clipped = np.maximum(eps, 0) for the non-linear terms
    # since plasticity only occurs for positive strain.
    eps_pos = np.maximum(eps, 0.0)
    
    # Calculate powers safely
    # If p3 or p5 is not integer, eps^p is undefined for negative eps.
    # Since we use eps_pos, it's fine.
    
    num_factor = 1.0 + p2 * np.power(eps_pos, p3)
    den_factor = 1.0 + p4 * np.power(eps_pos, p5) + p6 * np.power(T, p7)
    
    # Avoid division by zero
    den_safe = np.where(np.abs(den_factor) < 1e-8, 1e-8, den_factor)
    
    # Elastic term with non-linear modulation
    elastic_term = E_T * eps * (num_factor / den_safe)
    
    # Plastic saturation term: S(T) * tanh(eps / C)
    # We need a characteristic strain C. Let's fix it to a small value like 0.01
    # or derive it from params? We have no params left.
    # Let's use a fixed C = 0.01, which is typical for aluminum yield.
    C = 0.01
    S_T = p8 + p9 * T
    plastic_term = S_T * np.tanh(eps / C)
    
    sigma = elastic_term + plastic_term
    
    return sigma
