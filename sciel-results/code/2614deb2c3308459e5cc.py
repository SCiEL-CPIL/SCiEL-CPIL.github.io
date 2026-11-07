"""Symbolic regression: an equation for E. coli growth rate."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a biologically-motivated basis expansion.
    
    Combines the piecewise-linear "growth plateau" structure (trapezoidal windows
    for temperature and pH) from the high-performing inspiration with a richer set
    of interaction terms to capture non-separable effects.
    
    The model is linear in `params`, ensuring the optimization landscape is convex
    for the BFGS solver.
    
    Basis functions:
    1. Bias
    2. Substrate saturation (Monod)
    3. Temperature factor (trapezoidal)
    4. pH factor (trapezoidal)
    5. Density inhibition
    6. Substrate * Temperature interaction
    7. Substrate * pH interaction
    8. Temperature * pH interaction
    9. Substrate * Temperature * pH (full interaction)
    10. Substrate * Density interaction
    """
    # 1. Substrate saturation (Monod)
    # Using a standard Monod form s / (Ks + s). We fix Ks=1 for simplicity,
    # but the linear combination allows the magnitude to be fitted.
    u_s = s / (1.0 + s)
    
    # 2. Temperature response
    # Trapezoidal window: 0 below 30C, 1 between 35-40C, 0 above 45C.
    # This enforces strict zero-growth boundaries which is biologically plausible.
    t_factor = np.clip((temp - 30.0) / 5.0, 0.0, 1.0) * np.clip((45.0 - temp) / 5.0, 0.0, 1.0)
    
    # 3. pH response
    # Trapezoidal window: 0 below 6.0, 1 between 6.5-7.5, 0 above 8.0.
    p_factor = np.clip((pH - 6.0) / 0.5, 0.0, 1.0) * np.clip((8.0 - pH) / 0.5, 0.0, 1.0)
    
    # 4. Density inhibition
    # Rational term: 1 / (1 + b)
    d_factor = 1.0 / (1.0 + b)
    
    # Construct the linear combination of basis functions
    # params[0]: Bias
    # params[1]: Substrate effect (main driver)
    # params[2]: Temperature effect (independent, though likely small)
    # params[3]: pH effect (independent, though likely small)
    # params[4]: Density effect (independent)
    # params[5]: Substrate-Temp interaction (growth requires both substrate and viable temp)
    # params[6]: Substrate-pH interaction (growth requires both substrate and viable pH)
    # params[7]: Temp-pH interaction (co-optimality)
    # params[8]: Full multiplicative interaction (Substrate * Temp * pH) - captures synergy
    # params[9]: Substrate-Density interaction (carrying capacity effect)
    
    pred = (
        params[0] * 1.0 +
        params[1] * u_s +
        params[2] * t_factor +
        params[3] * p_factor +
        params[4] * d_factor +
        params[5] * (u_s * t_factor) +
        params[6] * (u_s * p_factor) +
        params[7] * (t_factor * p_factor) +
        params[8] * (u_s * t_factor * p_factor) +
        params[9] * (u_s * d_factor)
    )
    
    return pred
# EVOLVE-BLOCK-END
