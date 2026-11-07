"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Autonomous self-sustained oscillator with limit cycle behavior.
    
    Model: a = p1*x + p2*x^3 + p3*v + p4*v^3 + p5*x^2*v + p6*x*v + p7*x*v^2 + p8*x^2 + p9*v^2 + p10
    
    This implements an autonomous nonlinear oscillator model:
    - Duffing-type position nonlinearity: p1*x + p2*x^3
    - Van der Pol-style velocity terms: p3*v + p4*v^3 (negative damping + saturation)
    - Van der Pol damping cross-term: p5*x^2*v
    - Cross terms: p6*x*v, p7*x*v^2
    - Quadratic terms: p8*x^2, p9*v^2
    - Constant offset: p10
    """
    # Extract parameters
    p1 = params[0]
    p2 = params[1]
    p3 = params[2]
    p4 = params[3]
    p5 = params[4]
    p6 = params[5]
    p7 = params[6]
    p8 = params[7]
    p9 = params[8]
    p10 = params[9]
    
    # Precompute powers
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Duffing position terms
    pos_terms = p1 * x + p2 * x3
    
    # Velocity damping/saturation terms
    vel_terms = p3 * v + p4 * v3
    
    # Cross coupling terms
    cross_terms = p5 * x2 * v + p6 * x * v + p7 * x * v2
    
    # Quadratic terms
    quad_terms = p8 * x2 + p9 * v2
    
    # Constant
    bias = p10
    
    # Total acceleration
    a = pos_terms + vel_terms + cross_terms + quad_terms + bias
    
    return a
# EVOLVE-BLOCK-END
