# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-inspired Duffing oscillator + harmonic forcing + RBF correction.
    
    Structural change: replace the 3 RBFs in (x,v) space with 3 RBFs in (t,x,v) space,
    and add a second-harmonic forcing term (sin(2t), cos(2t)) to capture sub-harmonic
    response typical of nonlinear oscillators. The physics terms provide a well-conditioned
    landscape for BFGS starting from all-ones, while the RBFs capture residual structure
    that depends on the full state including time.
    
    Parameters:
    - p0: constant offset
    - p1: linear restoring force (x)
    - p2: linear damping (v)
    - p3: cubic stiffness (x^3)
    - p4: nonlinear damping (v*|v|)
    - p5: first-harmonic sin forcing
    - p6: first-harmonic cos forcing
    - p7: second-harmonic sin forcing
    - p8: second-harmonic cos forcing
    - p9: RBF correction weight (single RBF in 3D space)
    """
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Core Duffing oscillator dynamics
    nonlinear_damping = v * np.abs(v)
    
    result = (p0 
              + p1 * x 
              + p2 * v 
              + p3 * x * x * x 
              + p4 * nonlinear_damping 
              + p5 * np.sin(t) 
              + p6 * np.cos(t)
              + p7 * np.sin(2.0 * t) 
              + p8 * np.cos(2.0 * t))
    
    # RBF correction in 3D (t, x, v) space
    # Center at the middle of the domain
    t_center = 40.0
    x_center = -0.015  # midpoint of [-0.2056, 0.1749]
    v_center = 0.0117  # midpoint of [-0.3391, 0.3624]
    
    # Widths scaled to domain ranges
    t_width = 5.0
    x_width = 0.12
    v_width = 0.22
    
    dt = (t - t_center) / t_width
    dx = (x - x_center) / x_width
    dv = (v - v_center) / v_width
    
    dist_sq = dt * dt + dx * dx + dv * dv
    rbf = np.exp(-0.5 * dist_sq)
    
    result += p9 * rbf
    
    # Safety check
    if not np.all(np.isfinite(result)):
        return np.zeros_like(t)
    
    return result
# EVOLVE-BLOCK-END
