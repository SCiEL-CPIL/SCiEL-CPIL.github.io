"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator modeled as a rational function
    with a shared quadratic denominator to capture bounded nonlinear dynamics.
    
    y = (p0 + p1*x + p2*v + p3*x^2 + p4*v^2 + p5*x*v + p6*x^3 + p7*v^3) 
        / (1 + p8*x^2 + p9*v^2)
    """
    # Scale inputs to improve numerical conditioning
    # x range: [-0.5728, 0.5275], v range: [-0.1754, 0.2821]
    X = x * 2.0  # approximate scale to [-1, 1]
    V = v * 4.0  # approximate scale to [-1, 1]
    
    # Precompute terms
    X2 = X * X
    V2 = V * V
    XV = X * V
    X3 = X2 * X
    V3 = V2 * V
    
    # Numerator
    num = (params[0] + 
           params[1] * X + 
           params[2] * V + 
           params[3] * X2 + 
           params[4] * V2 + 
           params[5] * XV + 
           params[6] * X3 + 
           params[7] * V3)
    
    # Denominator (ensure it's well-behaved)
    den = 1.0 + params[8] * X2 + params[9] * V2
    
    # Safety: avoid division by zero or near-zero
    den = np.where(np.abs(den) < 1e-6, np.sign(den) * 1e-6, den)
    
    return num / den
# EVOLVE-BLOCK-END
