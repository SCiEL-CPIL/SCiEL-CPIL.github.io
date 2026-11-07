"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Strategy: Physics-Informed Structural Model (Van der Pol / Duffing Hybrid)
#
# The current approach uses a generic linear combination of basis functions 
# (polynomials, tanh, sin/cos) which treats the problem as a black-box 
# regression. This often leads to poor generalization if the basis functions 
# don't align with the underlying dynamics.
#
# This new strategy leverages the "Nonlinear Oscillator" hint. 
# A common model for such systems is:
# x'' = -omega_0^2 * x - mu * (x^2 - A^2) * x' - beta * x^3 + ...
#
# We model the acceleration (target y) as a function of x and v (velocity).
# We exclude explicit time dependence (t) because in an autonomous oscillator 
# on a limit cycle, the state (x, v) fully determines the acceleration. 
# Even if the system is driven, if the drive is periodic and the system 
# has settled into a periodic response, x and v are sufficient descriptors 
# of the phase. 
#
# The model includes:
# 1. Linear restoring force (Spring): p0 * x
# 2. Linear damping: p1 * v
# 3. Cubic stiffness (Duffing): p2 * x^3
# 4. Nonlinear damping (Van der Pol style): p3 * v * (1 + p4 * x^2) 
#    This term allows for negative damping at small amplitudes (energy injection) 
#    and positive damping at large amplitudes (saturation), which is characteristic 
#    of self-sustained oscillations.
# 5. Higher order cross terms to capture non-separability:
#    p5 * x * v
#    p6 * v^3
#    p7 * x^5
#    p8 * v * x^3
#    p9 * (x^2 + v^2) * v  (Energy-dependent damping)
#
# This structure is much more constrained and physically meaningful than a 
# generic polynomial basis, potentially allowing BFGS to find a better 
# global fit with fewer effective degrees of freedom and better generalization.

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-inspired closed-form expression for nonlinear oscillator acceleration.
    
    Parameters:
    t: Time (unused in this autonomous model, kept for interface)
    x: Position
    v: Velocity
    params: Array of 10 fitted constants [p0, ..., p9]
    
    Returns:
    y: Acceleration predictions
    """
    p = params
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = p
    
    # Pre-compute powers to avoid redundant calculation
    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    v2 = v * v
    v3 = v2 * v
    
    # Construct the acceleration model
    # Term 1: Linear spring
    term_1 = p0 * x
    
    # Term 2: Linear damping
    term_2 = p1 * v
    
    # Term 3: Cubic stiffness (Duffing)
    term_3 = p2 * x3
    
    # Term 4: Van der Pol-like nonlinear damping
    # Standard VdP: -mu * (1 - x^2) * v. Here we use a generalized form:
    # p3 * v * (1 + p4 * x^2)
    # If p4 is negative and large, it mimics (1 - k*x^2).
    term_4 = p3 * v * (1.0 + p4 * x2)
    
    # Term 5: Linear cross-coupling
    term_5 = p5 * x * v
    
    # Term 6: Cubic velocity (drag)
    term_6 = p6 * v3
    
    # Term 7: Quintic stiffness (hardening/softening refinement)
    term_7 = p7 * x5
    
    # Term 8: Velocity-Position cubic interaction
    term_8 = p8 * v * x3
    
    # Term 9: Energy-dependent damping
    term_9 = p9 * (x2 + v2) * v
    
    # Sum all terms
    y = term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8 + term_9
    
    return y

# EVOLVE-BLOCK-END
