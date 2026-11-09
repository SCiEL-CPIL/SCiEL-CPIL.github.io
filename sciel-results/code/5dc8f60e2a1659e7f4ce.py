"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Structurally different: a smooth, globally-defined basis-function model.
    Uses a logistic yield transition, a bounded hardening term, and
    temperature-dependent modulation of all key coefficients.
    """
    e = strain
    t = temp

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Temperature-dependent yield strain
    # t is in [0.06667, 1], center at 0.5
    t_centered = t - 0.5
    yield_strain = p2 + p6 * t_centered

    # Clamp yield_strain to a reasonable range to avoid extreme sigmoid values
    # This helps BFGS converge more reliably
    yield_strain = np.clip(yield_strain, -0.01, 0.5)

    # Smooth logistic transition from elastic to plastic
    # Sharpness controlled by p3
    # Use a safe sharpness to avoid overflow
    sharpness = np.clip(p3, 1.0, 100.0)
    x = sharpness * (e - yield_strain)
    # Numerically stable sigmoid
    transition = np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

    # Elastic modulus, temperature-dependent
    elastic_mod = p1 + p7 * t_centered

    # Elastic stress
    elastic_stress = elastic_mod * e

    # Plastic stress: yield point stress + hardening
    # Stress at yield point
    stress_at_yield = elastic_mod * yield_strain

    # Hardening term: bounded, smooth
    # plastic_strain = max(e - yield_strain, 0)
    # Use a smooth approximation: (e - yield_strain + sqrt((e-yield_strain)^2 + eps))/2
    # But simpler: just use the positive part, it's smooth enough for BFGS
    plastic_strain = np.maximum(e - yield_strain, 0.0)

    # Hardening: p4 * plastic_strain / (1 + p5 * plastic_strain)
    # This is a rational function that saturates, smoother than exp
    # But 1/(1+x) has a kink at x=0 if we use max. Let's use a different form.
    # Actually, let's use: p4 * plastic_strain * (1 - exp(-p5 * plastic_strain))
    # This is C1 continuous at plastic_strain=0
    # But exp can be problematic. Let's use the rational form with a smooth max.
    # smooth_max(x, 0) = (x + sqrt(x^2 + eps))/2
    eps = 1e-8
    smooth_plastic = 0.5 * (plastic_strain + np.sqrt(plastic_strain**2 + eps))
    # But this changes the behavior. Let's stick with the exponential form but make it safe.
    # For large positive x, exp(-p5*x) -> 0, so hardening -> p4 * plastic_strain
    # For x near 0, 1-exp(-p5*x) ~ p5*x, so hardening ~ p4*p5*x^2, which is smooth.
    
    # To avoid overflow in exp for very negative arguments (which won't happen since plastic_strain >= 0)
    # and underflow for very positive (exp -> 0, which is fine)
    exp_arg = -p5 * plastic_strain
    # Clip exp_arg to avoid underflow warnings, but np.exp handles it
    exp_term = np.exp(np.clip(exp_arg, -500, 500))
    hardening = p4 * plastic_strain * (1.0 - exp_term)

    # Temperature modulation of hardening
    hardening_mod = 1.0 + p8 * t_centered
    plastic_stress = stress_at_yield + hardening * hardening_mod

    # Blend elastic and plastic
    stress = (1.0 - transition) * elastic_stress + transition * plastic_stress

    # Thermal stress offset
    thermal = p9 * t_centered

    return p0 + stress + thermal
# EVOLVE-BLOCK-END
