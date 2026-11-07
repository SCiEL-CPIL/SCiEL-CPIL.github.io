"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature."""
    # params: length 10
    # p0: base elastic modulus (Young's modulus)
    # p1: temperature sensitivity of modulus (fractional change per temp unit)
    # p2: yield stress at reference temperature
    # p3: temperature sensitivity of yield stress
    # p4: hardening exponent (controls rate of strain hardening)
    # p5: saturation stress (ultimate strength at reference temp)
    # p6: temperature sensitivity of saturation stress
    # p7: width of yield transition (controls sharpness of elastic-to-plastic)
    # p8: compression correction factor
    # p9: residual/pre-stress offset

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Temperature-dependent elastic modulus
    # E(T) = p0 * (1 - p1 * temp)
    E = p0 * (1.0 - p1 * temp)

    # Temperature-dependent yield stress
    # sigma_y(T) = p2 * (1 - p3 * temp)
    sigma_y = p2 * (1.0 - p3 * temp)

    # Temperature-dependent saturation/ultimate stress
    # sigma_u(T) = p5 * (1 - p6 * temp)
    sigma_u = p5 * (1.0 - p6 * temp)

    # Handle tension and compression separately
    # For compression, use a modified model with p8 as correction factor
    # strain > 0: tension, strain < 0: compression

    # Smooth approximation of max(0, strain) and max(0, -strain)
    # Using softplus-like function for smoothness
    eps = 1e-8
    strain_pos = np.where(strain > 0, strain, eps * np.log1p(np.exp(-strain / eps)))
    strain_neg = np.where(strain < 0, -strain, eps * np.log1p(np.exp(strain / eps)))

    # Tension model:
    # Elastic region: stress = E * strain (for small strains)
    # Plastic region: stress = sigma_y + (sigma_u - sigma_y) * tanh(p4 * (strain - strain_y))
    # where strain_y is the yield strain

    # Yield strain: sigma_y / E
    E_safe = np.where(np.abs(E) > eps, E, eps)
    strain_y = sigma_y / E_safe

    # For tension:
    # Use a smooth interpolation between elastic and plastic response
    # t = smoothstep-like function that goes from 0 to 1 as strain exceeds yield
    # t = 1 / (1 + exp(-p7 * (strain - strain_y)))  -- sigmoid transition

    # Actually, let's use a cleaner formulation:
    # stress_tension = E * strain * (1 - t) + (sigma_y + (sigma_u - sigma_y) * tanh(p4 * (strain - strain_y))) * t
    # where t = sigmoid(p7 * (strain - strain_y))

    # But this might have issues. Let's use a different approach:
    # stress = E * strain + plastic_correction
    # where plastic_correction kicks in after yield

    # Better: use a smooth min/max approach
    # stress_tension = min(E * strain, sigma_y + (sigma_u - sigma_y) * tanh(p4 * max(0, strain - strain_y)))
    # But min is not smooth. Use smooth min via log-sum-exp.

    # Let's try a simpler but effective model:
    # stress_tension = E * strain * exp(-p4 * strain) + sigma_y * (1 - exp(-p4 * strain)) * (1 - exp(-p4 * (strain - strain_y)))
    # This is getting complicated. Let me use a well-known form.

    # Standard approach: Ramberg-Osgood type
    # stress = E * strain + sigma_y * (strain / strain_y)^(1/n) for strain > strain_y
    # But this is not smooth at yield.

    # Let's use the tanh-based model that worked well:
    # For tension (strain >= 0):
    #   elastic_part = E * strain
    #   plastic_part = sigma_y + (sigma_u - sigma_y) * tanh(p4 * (strain - strain_y))
    #   Use a smooth blend: stress = (1 - t) * elastic_part + t * plastic_part
    #   where t = 0.5 * (1 + tanh(p7 * (strain - strain_y)))

    # For compression (strain < 0):
    #   Similar but with p8 correction
    #   stress = E * strain * p8 (simple linear with correction)

    # Compute for tension
    t = 0.5 * (1.0 + np.tanh(p7 * (strain - strain_y)))
    elastic_part = E * strain
    plastic_part = sigma_y + (sigma_u - sigma_y) * np.tanh(p4 * (strain - strain_y))
    stress_tension = (1.0 - t) * elastic_part + t * plastic_part

    # Compute for compression
    stress_compression = E * strain * p8

    # Combine: use strain sign to select
    # Smooth selection: use tanh of strain
    sign_smooth = np.tanh(strain / eps)
    # When strain > 0, sign_smooth ~ 1, use tension
    # When strain < 0, sign_smooth ~ -1, use compression
    # weight_tension = 0.5 * (1 + sign_smooth)
    # weight_compression = 0.5 * (1 - sign_smooth)

    weight_tension = 0.5 * (1.0 + np.tanh(strain / 0.001))
    weight_compression = 0.5 * (1.0 - np.tanh(strain / 0.001))

    stress = weight_tension * stress_tension + weight_compression * stress_compression

    # Add residual stress
    stress = stress + p9

    return stress
# EVOLVE-BLOCK-END
