"""
Symbolic regression for stress-strain of an aluminium rod.
Uses a piecewise-linear approximation with ramp functions for efficient BFGS fitting.
The 10 parameters are fitted by the evaluator using BFGS.
"""
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model structure:
    - Piecewise-linear approximation using clipped linear terms (ramps)
    - Ramps allow modeling of yield points and hardening regimes
    - Temperature effects modify slopes and intercepts
    
    Parameters:
    p0: intercept
    p1: elastic slope (Young's modulus)
    p2: yield strain location (kink 1)
    p3: hardening strain location (kink 2)
    p4: plastic hardening slope
    p5: temperature scaling for elastic slope
    p6: temperature scaling for yield stress
    p7: temperature scaling for hardening slope
    p8: additional temperature interaction term
    p9: high-temperature offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Center temperature for better conditioning
    temp_centered = temp - 0.5
    
    # Temperature-dependent elastic slope
    E = p1 * (1.0 + p5 * temp_centered)
    
    # Temperature-dependent yield stress offset
    sigma_y_offset = p6 * temp_centered
    
    # Temperature-dependent hardening slope
    H = p4 * (1.0 + p7 * temp_centered)
    
    # Ramp functions: max(strain - kink, 0)
    # Kink 1: yield point
    # Kink 2: end of initial hardening / transition to uniform hardening
    # We can use multiple ramps to create a piecewise linear curve
    # Let's use 3 segments: elastic, initial hardening, uniform hardening
    # But we only have 10 params. Let's use 2 kinks.
    
    # Ramp 1: activates after strain > p2 (yield)
    # This changes the slope from E to E + p4 (or similar)
    # Actually, a standard piecewise linear model:
    # stress = p0 + E * strain + p4 * max(strain - p2, 0) + p8 * max(strain - p3, 0)
    # But we need to handle temperature.
    
    # Let's define:
    # stress = p0 + E(T) * strain + p4(T) * max(strain - p2, 0) + p8 * max(strain - p3, 0)
    # This is convex if E, p4, p8 are positive.
    # BFGS can handle this well.
    
    ramp1 = np.maximum(strain - p2, 0.0)
    ramp2 = np.maximum(strain - p3, 0.0)
    
    # Temperature effects on slopes
    E_T = p1 * (1.0 + p5 * temp_centered)
    H_T = p4 * (1.0 + p7 * temp_centered)
    
    # Base stress
    stress = p0 + E_T * strain + H_T * ramp1 + p8 * ramp2
    
    # Add temperature offset
    stress = stress + p6 * temp_centered + p9 * temp_centered * temp_centered
    
    return stress
