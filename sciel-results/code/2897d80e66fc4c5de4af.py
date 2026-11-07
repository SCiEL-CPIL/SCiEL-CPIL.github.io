"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a smooth piecewise model with elastic and plastic regimes:
    - Elastic regime: linear response with slope E
    - Plastic regime: hardening with a plateau
    - Temperature effects: thermal expansion and yield point reduction
    
    The model uses smooth hinge functions to avoid discontinuities
    while allowing independent fitting of elastic slope and yield plateau.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Elastic modulus (slope in elastic regime)
    E = p0
    
    # Yield stress (plateau level)
    Y = p1
    
    # Yield strain threshold (where plastic flow begins)
    eps_y = p2
    
    # Hardening rate after yield
    H = p3
    
    # Temperature coefficient for thermal strain
    alpha = p4
    
    # Temperature effect on yield stress
    beta = p5
    
    # Temperature effect on elastic modulus
    gamma = p6
    
    # Additional constant offset
    c0 = p7
    
    # Smoothing parameter for the hinge
    k = p8
    
    # Additional quadratic correction
    quad_coeff = p9
    
    # Temperature-dependent adjustments
    T = temp
    E_T = E * (1.0 + gamma * (T - 0.5))  # Normalize temp around 0.5
    Y_T = Y * (1.0 - beta * (T - 0.5))   # Yield decreases with temperature
    eps_y_T = eps_y + alpha * (T - 0.5)  # Yield strain shifts with temperature
    
    # Smooth hinge function: s * (1 - exp(-k * s))
    # This is ~ s for small s, and ~ 1/k for large s
    # We use this to create a smooth transition
    
    # Elastic component: linear up to yield, then saturates
    # Using smooth approximation of min(eps, eps_y)
    eps_clamped = np.where(strain > 0, strain, 0)
    
    # Smooth elastic part: E * smooth_min(eps, eps_y)
    # smooth_min(a, b) ≈ a - a * sigmoid((a-b)/k)
    # Or use: eps * (1 - exp(-k * (eps_y - eps))) for eps < eps_y
    # Better: use a smooth approximation
    
    # Let's use a different approach:
    # stress = E * eps * (1 - exp(-k * (eps_y - eps))) for eps < eps_y
    #         Y + H * (eps - eps_y) for eps > eps_y
    
    # Smooth version using tanh or similar
    
    # Compute elastic stress with saturation
    # For small eps: E * eps
    # For large eps: approaches E * eps_y
    # Use: E * eps_y * (1 - exp(-eps / eps_y)) but this doesn't give linear start
    
    # Better: use the smooth hinge directly
    # elastic_stress = E * smooth_hinge(strain, eps_y)
    # where smooth_hinge(x, a) = x for x < a, and a for x > a (smoothly)
    
    # smooth_hinge(x, a) = a * (1 - exp(-(x - a) / k)) for x > a
    #                      x for x < a
    # This is discontinuous in derivative at x=a
    
    # Let's use a different formulation:
    # stress = E * eps + (Y - E * eps_y) * smoothstep(eps, eps_y, k)
    # where smoothstep transitions from 0 to 1
    
    # smoothstep(x, a, k) = 0.5 * (1 + tanh(k * (x - a)))
    
    # But we want the slope to change, not just add a step
    
    # Physical model:
    # In elastic regime: stress = E * eps
    # In plastic regime: stress = Y + H * (eps - eps_y)
    # At yield: E * eps_y = Y (for continuity)
    
    # Smooth interpolation:
    # stress = (1 - w) * E * eps + w * (Y + H * (eps - eps_y))
    # where w = smoothstep(eps, eps_y, k)
    
    # But this may not be continuous if E * eps_y != Y
    
    # Let's enforce continuity by construction:
    # Let Y = E * eps_y (yield stress equals elastic stress at yield strain)
    # Then in plastic: stress = E * eps_y + H * (eps - eps_y)
    
    # So: stress = E * eps + H * (eps - eps_y) * w(eps)
    # where w(eps) = 0 for eps < eps_y, 1 for eps > eps_y (smooth)
    
    # This ensures continuity and differentiability
    
    w = 0.5 * (1.0 + np.tanh(k * (strain - eps_y_T)))
    
    # Base elastic stress
    stress_elastic = E_T * strain
    
    # Plastic hardening term
    stress_plastic = H * (strain - eps_y_T) * w
    
    # Total stress
    stress = stress_elastic + stress_plastic
    
    # Add thermal effects and corrections
    stress = stress + c0 + quad_coeff * strain**2
    
    return stress
# EVOLVE-BLOCK-END
