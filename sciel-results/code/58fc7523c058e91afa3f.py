"""Stress-strain model for aluminium rod with temperature dependence."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a bilinear elastic-plastic model with smooth tanh transition
    and power-law hardening, all with temperature dependence.
    
    Model:
        E(T)   = p0 + p1*T       (elastic modulus)
        s_y(T) = p2 + p3*T       (yield strain)
        Y(T)   = p4 + p5*T       (yield stress level)
        H(T)   = p6 + p7*T       (hardening modulus)
        n      = p8              (hardening exponent, clamped to [0.5, 3.0])
        k      = p9              (transition sharpness)
        
        step = 0.5 * (1 + tanh(k * (strain - s_y) / max(|s_y|, 1e-10)))
        
        elastic = E(T) * strain * (1 - step)
        plastic = Y(T) + H(T) * max(strain - s_y, 0)^n * step
        
        stress = elastic + plastic
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent parameters
    E = p0 + p1 * temp
    s_y = p2 + p3 * temp
    Y = p4 + p5 * temp
    H = p6 + p7 * temp
    n = np.clip(p8, 0.5, 3.0)
    k = p9
    
    # Normalize the transition variable to avoid scale issues
    scale = np.maximum(np.abs(s_y), 1e-10)
    x = (strain - s_y) / scale
    
    # Smooth step function using tanh
    # Clamp x to prevent tanh saturation issues (tanh is bounded, but large args can lose precision)
    x_clamped = np.clip(x, -50.0, 50.0)
    step = 0.5 * (1.0 + np.tanh(k * x_clamped))
    
    # Elastic contribution (dominant before yield)
    elastic = E * strain * (1.0 - step)
    
    # Plastic contribution (dominant after yield)
    # Use max to ensure non-negative base for power
    excess = np.maximum(strain - s_y, 0.0)
    # For the power term, use a small epsilon to avoid 0^0 issues and improve gradient
    excess_safe = np.maximum(excess, 1e-15)
    hardening = H * np.power(excess_safe, n)
    plastic = Y + hardening
    
    # Combine: before yield, stress ≈ elastic; after yield, stress ≈ plastic
    stress = elastic + plastic * step
    
    return stress
# EVOLVE-BLOCK-END
