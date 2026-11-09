# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using Monod kinetics for substrate,
    and quadratic terms for temperature and pH to capture optimal ranges.
    Includes interaction terms between temperature and pH.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Normalize inputs to improve numerical stability for BFGS
    # b: population density [1, 5] -> center 3, scale 2
    # s: substrate [1, 40] -> center 10, scale 10 (or use Monod directly)
    # temp: [1, 40] -> center 20, scale 10
    # pH: [1, 14] -> center 7, scale 3

    # Substrate effect: Monod kinetics
    # mu_s = s / (Ks + s)
    # We fit Ks as part of the model structure? No, params are just coefficients.
    # We need a fixed form or use params to shape it.
    # Let's use a generic saturating function: s / (p0 + s)
    # But p0 is one of the 10 params. If p0 is negative, we have issues.
    # Better: use a fixed Ks? No, we want to fit it.
    # Let's try: s_term = s / (p0 + s) where p0 is fitted.
    # To ensure positivity of denominator, we can use exp(p0) or similar, but that changes gradient.
    # Alternatively, just use polynomial in s? The Monod is biologically motivated.
    # Let's assume p0 > 0 by initialization (starts at 1) and hope BFGS keeps it positive.
    # If p0 becomes negative, s/(p0+s) might blow up.
    # Safer: use s / (p0**2 + s) ? No, that's not standard.
    # Let's stick to a simple polynomial for s as well, or a robust Monod.
    # Given the small number of params (10), let's allocate:
    # 1 param for Monod Ks (p0)
    # 1 param for max growth rate scaling (p1)
    # 2 params for temp quadratic (p2, p3)
    # 2 params for pH quadratic (p4, p5)
    # 1 param for temp-pH interaction (p6)
    # 1 param for density effect (p7) - maybe inhibitory?
    # 1 param for density-substrate interaction? (p8)
    # 1 bias (p9)

    # Let's refine the structure:
    # mu = p1 * (s / (p0 + s)) * (1 + p2*(temp - 20) + p3*(temp - 20)**2) * (1 + p4*(pH - 7) + p5*(pH - 7)**2) * (1 + p6*(temp-20)*(pH-7)) * (1 - p7*b) + p8*b*s + p9
    
    # This uses 10 params.
    # p0: Ks (half-saturation constant)
    # p1: max growth rate scaling
    # p2, p3: temp linear and quadratic coefficients (normalized)
    # p4, p5: pH linear and quadratic coefficients (normalized)
    # p6: temp-pH interaction
    # p7: density inhibition coefficient
    # p8: density-substrate interaction (or just another term)
    # p9: bias

    # Normalize temp and pH to center around their means to make quadratic coefficients more interpretable and stable
    temp_center = 20.0
    ph_center = 7.0
    t = temp - temp_center
    h = pH - ph_center

    # Monod term
    # Ensure denominator is not zero. p0 starts at 1.
    # If p0 becomes small, s/(p0+s) is fine. If p0 becomes negative, it can be problematic.
    # We can clip p0 to a small positive value? No, that breaks gradients.
    # Let's trust BFGS to stay in reasonable bounds if the landscape is nice.
    # Or use a safer form: s / (p0 + s) where p0 is exp of a param? 
    # If we use exp, the param is log(Ks). Let's do that for safety.
    Ks = np.exp(p0)
    monod = s / (Ks + s)

    # Temp and pH effects
    temp_effect = 1.0 + p2 * t + p3 * t**2
    ph_effect = 1.0 + p4 * h + p5 * h**2
    interaction = 1.0 + p6 * t * h

    # Density effect
    # Often high density inhibits growth.
    # Let's use a simple linear inhibition: (1 - p7 * b)
    # Or maybe b is just a covariate.
    # Let's include a term for b directly.
    density_term = 1.0 - p7 * b

    # Combined
    # mu = p1 * monod * temp_effect * ph_effect * interaction * density_term + p8 * b * s + p9
    
    # Wait, p8 * b * s might be redundant or conflicting.
    # Let's simplify.
    # mu = p1 * monod * temp_effect * ph_effect * interaction * density_term + p9
    
    # That uses p0..p7, p9. p8 is unused.
    # Let's use p8 for a second order density term or interaction with temp.
    # density_term = 1.0 - p7 * b + p8 * b**2
    
    # Let's try:
    density_term = 1.0 - p7 * b
    
    # We have one param left (p8). Let's use it for an interaction between substrate and temp?
    # Or just add it to the bias? No.
    # Let's add a term: p8 * s * t (substrate-temp interaction)
    
    base = p1 * monod * temp_effect * ph_effect * interaction * density_term
    extra = p8 * s * t
    
    result = base + extra + p9

    return result
# EVOLVE-BLOCK-END
