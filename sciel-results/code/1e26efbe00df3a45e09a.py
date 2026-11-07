# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid physics-inspired model + RBF correction for nonlinear oscillator.
    
    The 10 parameters are split:
    - p0: constant offset
    - p1: linear restoring force coefficient (x)
    - p2: damping coefficient (v)
    - p3: cubic stiffness (x^3)
    - p4: quadratic damping (v^2 * sign(v))
    - p5: forcing amplitude sin(t)
    - p6: forcing amplitude cos(t)
    - p7-p9: RBF correction weights (3 RBFs)
    
    This structure gives BFGS a well-conditioned landscape because the physics
    terms are smooth and meaningful, while RBFs capture residual structure.
    """
    # Physics-inspired terms
    p0 = params[0]  # offset
    p1 = params[1]  # linear restoring
    p2 = params[2]  # damping
    p3 = params[3]  # cubic stiffness
    p4 = params[4]  # nonlinear damping
    p5 = params[5]  # sin forcing
    p6 = params[6]  # cos forcing
    p7 = params[7]  # RBF 1 weight
    p8 = params[8]  # RBF 2 weight
    p9 = params[9]  # RBF 3 weight
    
    # Core Duffing oscillator dynamics
    # y = a0 + a1*x + a2*v + a3*x^3 + a4*v*|v| + a5*sin(t) + a6*cos(t)
    nonlinear_damping = v * np.abs(v)
    result = (p0 
              + p1 * x 
              + p2 * v 
              + p3 * x * x * x 
              + p4 * nonlinear_damping 
              + p5 * np.sin(t) 
              + p6 * np.cos(t))
    
    # RBF correction terms with fixed centers in (x, v) space
    # These capture residual structure not explained by the physics model
    # Center 1: near origin
    dx1 = (x - 0.0) / 0.15
    dv1 = (v - 0.0) / 0.25
    rbf1 = np.exp(-0.5 * (dx1 * dx1 + dv1 * dv1))
    
    # Center 2: positive x, negative v
    dx2 = (x - 0.1) / 0.15
    dv2 = (v + 0.15) / 0.25
    rbf2 = np.exp(-0.5 * (dx2 * dx2 + dv2 * dv2))
    
    # Center 3: negative x, positive v
    dx3 = (x + 0.1) / 0.15
    dv3 = (v - 0.15) / 0.25
    rbf3 = np.exp(-0.5 * (dx3 * dx3 + dv3 * dv3))
    
    result += p7 * rbf1 + p8 * rbf2 + p9 * rbf3
    
    # Safety check
    if not np.all(np.isfinite(result)):
        return np.zeros_like(t)
    
    return result
# EVOLVE-BLOCK-END
