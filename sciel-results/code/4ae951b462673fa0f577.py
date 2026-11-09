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
    p1: slope for strain (elastic region)
    p2: strain knot point 1 (yield point)
    p3: slope after knot 1
    p4: strain knot point 2 (hardening point)
    p5: slope after knot 2
    p6: temperature scaling for intercept
    p7: temperature scaling for elastic slope
    p8: temperature scaling for yield strain
    p9: temperature scaling for hardening slope
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Center temperature for better conditioning (range ~0.067 to 1.0)
    temp_c = temp - 0.533
    
    # Temperature-dependent parameters
    intercept = p0 + p6 * temp_c
    elastic_slope = p1 * (1.0 + p7 * temp_c)
    yield_strain = p2 + p8 * temp_c
    hardening_slope = p3 * (1.0 + p9 * temp_c)
    
    # Piecewise-linear stress-strain curve using ramp functions
    # Ramp function: max(0, x) creates a kink at x=0
    # This gives us a smooth, differentiable approximation of the yield point
    
    # Elastic region: linear with slope elastic_slope
    # After yield: reduced slope due to plastic deformation
    
    # Using ramp functions to create piecewise-linear behavior
    # stress = intercept + elastic_slope * strain + (hardening_slope - elastic_slope) * max(0, strain - yield_strain)
    
    ramp1 = np.maximum(0.0, strain - yield_strain)
    
    # Additional ramp for more complex hardening behavior if needed
    # But keep it simple with just one kink for now
    
    stress = intercept + elastic_slope * strain + (hardening_slope - elastic_slope) * ramp1
    
    return stress
