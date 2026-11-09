"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a multiplicative temperature-modulated elastic-plastic model:
    - Temperature-dependent elastic modulus: E(T) = E0 * (1 + a*T + b*T^2)
    - Temperature-dependent yield stress: sigma_y(T) = sy0 * (1 + c*T + d*T^2)
    - Smooth hyperbolic sine saturation for plastic hardening
    - Multiplicative coupling between temperature and plasticity
    
    10 parameters:
    p0: intercept (thermal stress offset)
    p1: Young's modulus E0
    p2: yield stress sy0
    p3: yield strain location
    p4: temperature coefficient for E (linear)
    p5: temperature coefficient for E (quadratic)
    p6: temperature coefficient for sy (linear)
    p7: temperature coefficient for sy (quadratic)
    p8: hardening saturation scale
    p9: hardening amplitude
    """
    # Ensure inputs are arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent elastic modulus
    E_temp = p1 * (1.0 + p4 * temp + p5 * temp * temp)
    
    # Temperature-dependent yield stress
    sy_temp = p2 * (1.0 + p6 * temp + p7 * temp * temp)
    
    # Yield strain (where plastic deformation starts)
    yield_strain = p3
    
    # Smooth transition from elastic to plastic using tanh
    # Transition width based on hardening scale
    transition_width = 1.0 / (1.0 + np.abs(p8))
    elastic_plastic_transition = 0.5 * (1.0 + np.tanh((strain - yield_strain) / transition_width))
    
    # Elastic part: active before yield
    elastic_part = E_temp * strain * (1.0 - elastic_plastic_transition)
    
    # Plastic strain
    plastic_strain = strain - yield_strain
    plastic_strain_pos = np.clip(plastic_strain, 0.0, None)
    
    # Hyperbolic sine saturation for hardening
    # sinh(x) grows exponentially, but we use a scaled version to create saturation
    # The saturation level is p9, and the scale is p8
    hardening_saturation = p9 * np.tanh(plastic_strain_pos * (1.0 + np.abs(p8)))
    
    # Plastic part with temperature-modulated yield and hardening
    plastic_part = elastic_plastic_transition * (sy_temp + hardening_saturation)
    
    # Combine all terms
    prediction = p0 + elastic_part + plastic_part
    
    # Ensure finite output
    prediction = np.where(np.isfinite(prediction), prediction, 0.0)
    
    return prediction
# EVOLVE-BLOCK-END
