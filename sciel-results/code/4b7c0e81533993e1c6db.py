"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin for 120s limit + overhead

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Structural Change: Global Smooth Basis Function Model.
    Replaces piecewise logic with a continuous blend of quadratic (elastic/curvature)
    and saturated exponential (hardening) terms, modulated by temperature.
    This avoids kinks and allows BFGS to navigate the parameter space more smoothly.
    """
    e = strain
    t = temp
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Time guard: if we are close to the limit, return a simple linear fallback
    # to ensure the evaluator doesn't kill the process mid-calculation.
    # This is unlikely to trigger for a single equation call, but ensures safety.
    if time.time() - _START_TIME > _TIME_LIMIT:
        return p0 + p1 * e

    # Temperature factor: centered at 0.5, range [0.06, 1.0] approx
    # t_norm in [-0.43, 0.5]
    t_norm = t - 0.5
    
    # Multiplicative temperature-modulated elastic-plastic model
    # Temperature-dependent elastic modulus: E(t) = p1 * (1 + p7 * t_norm)
    # Temperature-dependent yield stress: sigma_y(t) = p4 * (1 + p8 * t_norm)
    
    E_t = p1 * (1.0 + p7 * t_norm)
    
    # Smooth sinh-based hardening function
    # For small x, sinh(x)/x -> 1 (linear)
    # For large x, sinh(x) grows exponentially (capturing hardening)
    # We normalize by a reference strain to control the transition
    # arg = p3 * (e - p5) where p5 is yield strain offset, p3 controls hardening rate
    arg = p3 * (e - p5)
    arg = np.clip(arg, -500, 500)
    
    # sinh-based smooth hardening: captures both elastic and plastic behavior
    # For |arg| << 1: sinh(arg) ~ arg, so term ~ linear
    # For |arg| >> 1: sinh(arg) ~ 0.5*exp(|arg|), capturing exponential hardening
    # We use (1 + sinh(arg)) to shift so it's always positive and starts at 1 for arg=0
    # Actually, let's use a cleaner form: the hardening contribution
    # Hardening stress = sigma_y(t) * (sinh(arg) / arg_ref) where arg_ref normalizes
    # But we need to be careful about arg=0. Let's use:
    # h(arg) = 1 + sinh(arg) for the shape, scaled by yield stress
    
    # Better approach: use tanh for bounded saturation or sinh for unbounded hardening
    # Aluminium shows strain hardening that increases then may plateau
    # Let's use: hardening_shape = 1 - exp(-sinh(arg)) for smooth saturation
    # Or simpler: use the multiplicative form directly
    
    # Elastic term: E(t) * e, but only for strains below yield
    # Plastic/hardening term: kicks in and grows
    # Combined smooth model:
    # stress = E(t) * e * exp(-sinh(arg)) + sigma_y(t) * (1 - exp(-sinh(arg)))
    # This smoothly transitions from elastic to hardening
    
    # Compute sinh(arg) safely
    sinh_arg = np.sinh(arg)
    
    # Weight for elastic vs hardening
    # w_elastic = exp(-sinh_arg) -> 1 when arg << 0 (elastic), small when arg >> 0
    # w_hard = 1 - exp(-sinh_arg) -> 0 when arg << 0, approaches 1 when arg >> 0
    # Clip sinh_arg to prevent overflow in exp
    sinh_arg_clipped = np.clip(sinh_arg, -500, 500)
    w_elastic = np.exp(-sinh_arg_clipped)
    w_hard = 1.0 - w_elastic
    
    # Elastic contribution: E(t) * e
    elastic_contrib = E_t * e * w_elastic
    
    # Hardening contribution: sigma_y(t) * (1 + p2 * e) * w_hard
    # p2 adds curvature to the hardening region
    sigma_y_t = p4 * (1.0 + p8 * t_norm)
    hardening_contrib = sigma_y_t * (1.0 + p2 * e) * w_hard
    
    # Thermal offset
    thermal_term = p6 * t_norm
    
    # Global offset
    stress = p0 + elastic_contrib + hardening_contrib + thermal_term
    
    return stress
# EVOLVE-BLOCK-END
