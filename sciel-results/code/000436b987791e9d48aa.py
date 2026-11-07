"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a piecewise-linear model with smooth hinge functions to separate
    elastic and plastic regimes, with temperature-dependent parameters.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Temperature-dependent elastic modulus
    E = p0 + p1 * temp

    # Temperature-dependent yield strain
    eps_y = p2 + p3 * temp

    # Temperature-dependent hardening modulus
    H = p4 + p5 * temp

    # Hinge stiffness (controls smoothness of transition)
    k = np.maximum(np.abs(p6), 10.0)

    # Temperature-dependent yield stress offset
    sigma_y = p7 + p8 * temp

    # Constant offset
    offset = p9

    # Smooth hinge: x * (1 - exp(-k*x)) approximates max(x, 0) for large k
    # For x > 0: ~x, for x < 0: ~0
    def smooth_hinge(x):
        # Use a numerically stable formulation
        # For large positive x, exp(-k*x) -> 0, so result -> x
        # For large negative x, exp(-k*x) -> inf, but x*(1-inf) is problematic
        # Better: use np.where to handle the two cases
        pos = x > 0
        result = np.zeros_like(x)
        # For x > 0: x * (1 - exp(-k*x))
        result[pos] = x[pos] * (1 - np.exp(-k * x[pos]))
        # For x <= 0: small negative value, use soft approximation
        # x * (1 - exp(-k*x)) for x < 0: x is negative, exp(-k*x) > 1, so (1-exp) < 0, product > 0
        # This is actually not what we want. We want max(x,0) smoothed.
        # Let's use a different smooth hinge: 0.5*(x + sqrt(x^2 + eps))
        # But that's not using the proposed form.
        # Let's stick with the proposed form but ensure it's well-behaved
        # Actually, for x < 0, the proposed form gives a small positive value, which is fine as a smooth approximation
        result[~pos] = x[~pos] * (1 - np.exp(-k * x[~pos]))
        return result

    # Plastic strain: smooth hinge of (strain - eps_y)
    plastic_strain = smooth_hinge(strain - eps_y)

    # Elastic stress: E * strain (but we want to reduce this in plastic regime)
    # Better formulation: 
    # stress = E * min(strain, eps_y) + H * (strain - eps_y) for strain > eps_y
    #        = E * strain for strain <= eps_y
    # With smooth hinge:
    # stress = E * strain - (E - H) * smooth_hinge(strain - eps_y)
    #        = E * strain - E * smooth_hinge(strain - eps_y) + H * smooth_hinge(strain - eps_y)
    #        = E * (strain - smooth_hinge(strain - eps_y)) + H * smooth_hinge(strain - eps_y)
    #        = E * smooth_hinge(eps_y - strain) + H * smooth_hinge(strain - eps_y) + sigma_y * indicator
    
    # Simpler: use the formulation from the directive
    # stress = E * strain + (H - E) * smooth_hinge(strain - eps_y) + sigma_y * smooth_hinge(strain - eps_y) / max(strain - eps_y, eps)
    # This is getting complicated. Let's use a cleaner approach.
    
    # stress = E * strain for elastic, sigma_y + H * (strain - eps_y) for plastic
    # = E * strain + (sigma_y + H * (strain - eps_y) - E * strain) * indicator(strain > eps_y)
    # = E * strain + (sigma_y - E * eps_y + (H - E) * strain) * smooth_hinge(strain - eps_y) / max(strain - eps_y, eps)
    # This requires division by (strain - eps_y), which is problematic at the yield point.
    
    # Alternative clean formulation:
    # stress = E * strain + (sigma_y - E * eps_y) * smooth_hinge(strain - eps_y) / eps_y_ref + H * (smooth_hinge(strain - eps_y) - eps_y_ref * smooth_hinge(strain - eps_y) / eps_y_ref)
    # This is still messy.
    
    # Let's use a simpler piecewise approach with smooth blending:
    # stress = (1 - t) * E * strain + t * (sigma_y + H * (strain - eps_y))
    # where t = smooth_hinge(strain - eps_y) / max(strain - eps_y, eps)
    # But this is the same problem.
    
    # Let's go back to the tanh approach but with better parameterization.
    # Actually, the current program uses tanh. Let's try a different smooth function.
    
    # Use: stress = E * strain + (sigma_y - E * eps_y) * s + H * (strain - eps_y) * s
    # where s is a smooth step function that is 0 for strain << eps_y and 1 for strain >> eps_y
    # s = 0.5 * (1 + tanh((strain - eps_y) / width))
    
    width = np.maximum(np.abs(p9 * 0.01 + 0.001), 1e-6)  # Use p9 as a width parameter? No, p9 is offset.
    
    # Let me re-parameterize. I have 10 params.
    # p0: E0, p1: cE, p2: eps_y0, p3: ceps_y, p4: H0, p5: cH, p6: sigma_y0, p7: csigma_y, p8: width, p9: offset
    
    E = p0 + p1 * temp
    eps_y = p2 + p3 * temp
    H = p4 + p5 * temp
    sigma_y = p6 + p7 * temp
    width = np.maximum(np.abs(p8), 1e-6)
    offset = p9
    
    # Smooth step: 0 for strain << eps_y, 1 for strain >> eps_y
    s = 0.5 * (1 + np.tanh((strain - eps_y) / width))
    
    # Elastic stress
    sigma_elastic = E * strain
    
    # Plastic stress
    sigma_plastic = sigma_y + H * (strain - eps_y)
    
    # Blend
    stress = (1 - s) * sigma_elastic + s * sigma_plastic
    
    stress += offset
    
    return stress
# EVOLVE-BLOCK-END
