"""Bootstrap portfolio: model-and-optimize for stress-strain of aluminium rod."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We keep the function signature: equation(strain, temp, params) -> np.ndarray
# The evaluator will fit `params` (length 10) using BFGS from all ones.
# We need to design a closed-form expression with 10 parameters that can capture
# the stress-strain curve of aluminium, which typically has:
# 1. Elastic region (linear)
# 2. Yield point
# 3. Plastic region (strain hardening, possibly power-law or exponential)
# 4. Temperature effects (softening at higher temperatures)

# A common model for metals is:
# stress = E * strain * (1 - strain/strain_max) for small strains (elastic with some hardening)
# or use a more flexible form like:
# stress = E * strain / (1 + A * strain^2)  (hyperbolic approximation)
# Or: stress = E * strain * exp(-B * strain)  (exponential decay of stiffness)

# Let's use a flexible model that can capture:
# - Linear elastic behavior at small strain
# - Strain hardening or saturation at large strain
# - Temperature dependence

# Model: stress = f(strain, temp)
# Let's try: stress = (E_eff * strain) / (1 + k * strain^n) + T_effect
# where E_eff depends on temperature

# More specifically, let's use:
# stress = (p0 * (p1 + p2*temp) * strain) / (1 + p3 * strain^p4) + p5*temp + p6
# plus additional terms for better fitting

# Actually, let's use a more standard approach for metal stress-strain:
# True stress-strain often follows: sigma = K * epsilon^n (Hollomon)
# But engineering stress has a different shape.

# Let's use a combination that's flexible:
# stress = p0 * strain * (1 + p1*strain) / (1 + p2*strain^2 + p3*strain^4) 
#         * (1 + p4*temp + p5*temp^2)
#         + p6*strain + p7*temp + p8

# That's 9 parameters. Let's add one more for a shift or additional nonlinearity.

# Better approach: use a rational function that can fit various curves.

# Let's define:
# stress = (p0 + p1*strain + p2*strain^2 + p3*temp + p4*temp^2 + p5*strain*temp)
#        / (1 + p6*strain + p7*strain^2 + p8*temp + p9*strain*temp)

# This is a bivariate rational function with 10 parameters. It's very flexible
# and can approximate many smooth functions. The denominator ensures it doesn't
# blow up if parameters are reasonable.

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.
    
    Uses a bivariate rational function model with 10 fitted parameters.
    Normalized inputs for better BFGS conditioning.
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to roughly [-1, 1] range for better conditioning
    # strain in [-0.001911, 0.9636] -> center ~0.48, scale ~0.48
    s = (strain - 0.48) / 0.48
    # temp in [0.06667, 1] -> center ~0.53, scale ~0.47
    t = (temp - 0.53) / 0.47
    
    # Numerator: flexible polynomial in normalized coords
    num = (p0 + p1 * s + p2 * s**2 + 
           p3 * t + p4 * t**2 + 
           p5 * s * t)
    
    # Denominator: 1 + small perturbation terms (keep near 1 for stability)
    # Use tanh to keep denominator bounded and smooth
    den = 1.0 + p6 * s + p7 * s**2 + p8 * t + p9 * s * t
    
    # Safety: clamp denominator away from zero
    den_safe = np.where(np.abs(den) < 0.05, np.sign(den + 1e-300) * 0.05, den)
    
    stress = num / den_safe
    
    return stress
# EVOLVE-BLOCK-END
