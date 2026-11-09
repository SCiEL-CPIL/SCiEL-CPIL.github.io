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
    
    Uses a physically-motivated model combining linear elasticity and 
    power-law strain hardening (Hollomon-like) with temperature dependence.
    """
    # Ensure inputs are 1D arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent Elastic Modulus E(T)
    E = p0 * (1.0 + p1 * temp)
    
    # Temperature-dependent Flow Stress Coefficient K(T)
    K = p2 * (1.0 + p3 * temp)
    
    # Temperature-dependent Hardening Exponent n(T)
    n = p4 + p5 * temp
    
    # Absolute strain for the power-law term
    abs_strain = np.abs(strain)
    
    # Plastic term: K * |eps|^n / (1 + alpha * |eps|)
    denom = 1.0 + p6 * abs_strain
    
    # Avoid division by zero in denominator
    den_safe = np.where(np.abs(denom) < 1e-10, 1e-10 * np.sign(denom + 1e-300), denom)
    
    # Use a small offset in base to handle potential n < 0 gracefully (though unlikely)
    # and ensure 0**0 is not an issue (0**0 is 1 in numpy, but 0**n is 0 for n>0)
    # If n is negative, we get inf. 
    # To prevent inf from killing the fit, we can clip the power result?
    # No, let's just use abs_strain. If it fails, it fails.
    # But 0 ** negative is inf. 
    # Let's add a tiny epsilon to abs_strain to prevent inf if n < 0.
    safe_strain = abs_strain + 1e-8
    
    plastic_term = K * (safe_strain ** n) / den_safe
    
    # Elastic term
    elastic_term = E * strain
    
    # Total stress
    stress = elastic_term + plastic_term
    
    # Additional corrections
    stress = stress + p7 * strain + p8 * temp + p9
    
    return stress
# EVOLVE-BLOCK-END
