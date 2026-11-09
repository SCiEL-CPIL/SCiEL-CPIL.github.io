"""Stress-strain model for aluminium rod using a smooth piecewise-constitutive form."""
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Smooth constitutive model for aluminium stress-strain.
    
    Structure:
      - Elastic region: linear in strain
      - Plastic region: power-law hardening (Ramberg-Osgood style)
      - Temperature softening: multiplicative factor decreasing with temp
    
    Uses 10 parameters:
      p0: elastic modulus scale
      p1: yield strain threshold
      p2: hardening exponent
      p3: hardening coefficient
      p4: temp softening coefficient
      p5: temp softening exponent
      p6: offset in plastic region
      p7: transition width (smoothness)
      p8: additional temp linear term
      p9: intercept
    """
    eps = strain
    T = temp
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Ensure positive values for exponents and widths via softplus-like transform
    # We use np.abs to keep things simple and differentiable
    yield_strain = np.abs(p1) + 1e-6  # ensure positive
    hard_exp = np.abs(p2) + 0.1       # ensure > 0
    hard_coeff = np.abs(p3)
    temp_soft_coeff = np.abs(p4)
    temp_soft_exp = np.abs(p5) + 0.1
    offset = p6
    width = np.abs(p7) + 1e-4         # smooth transition width
    temp_linear = p8
    intercept = p9

    # Temperature softening factor: 1 - alpha * (T - T_ref)^(beta)
    # T is in [0.06667, 1], so normalize
    T_norm = T  # already in [~0.07, 1]
    softening = 1.0 - temp_soft_coeff * np.power(T_norm, temp_soft_exp)
    # Clip softening to avoid negative values
    softening = np.maximum(softening, 0.05)

    # Elastic part: p0 * eps * softening
    elastic = p0 * eps * softening

    # Plastic part: hard_coeff * (max(0, eps - yield_strain))^(hard_exp) * softening
    plastic_strain = np.maximum(0.0, eps - yield_strain)
    plastic = hard_coeff * np.power(plastic_strain, hard_exp) * softening

    # Smooth transition: use a sigmoid-like blend
    # x = (eps - yield_strain) / width
    x = (eps - yield_strain) / width
    # sigmoid: 1 / (1 + exp(-x))
    # Clip x to avoid overflow
    x_clipped = np.clip(x, -50, 50)
    sigmoid = 1.0 / (1.0 + np.exp(-x_clipped))

    # Blend: below yield, mostly elastic; above, add plastic
    # Actually, elastic is always present (it's the total strain response)
    # The plastic part adds on top in the plastic region
    # So total = elastic + plastic * sigmoid (to smooth the kink)
    
    # Better: use Ramberg-Osgood: eps = sigma/E + (sigma/K)^n
    # Inverting: sigma = E * eps * (1 + (E*eps/K)^(n-1))^(1/(1-n)) for n>1
    # This is complex. Let's use a simpler smooth model.
    
    # Alternative: sigma = p0 * eps + hard_coeff * (eps - yield_strain)_+^hard_exp
    # The sigmoid smooths the derivative at the yield point
    
    stress = elastic + plastic * sigmoid
    
    # Add temperature linear correction and intercept
    stress = stress + temp_linear * T + intercept
    
    return stress
