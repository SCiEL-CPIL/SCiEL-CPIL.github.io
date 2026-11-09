"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator modeled as a Generalized Duffing-van der Pol system.
    
    The equation is structured to capture the physics of a driven or self-excited oscillator:
    a = -k_eff(x) * x - c_eff(x, v) * v
    
    Where:
    - k_eff(x) is an effective stiffness that can be nonlinear (Duffing term).
    - c_eff(x, v) is an effective damping coefficient that can depend on state (van der Pol term).
    
    We use 10 parameters to define these functions:
    p[0]: Linear stiffness (k)
    p[1]: Cubic stiffness coefficient (beta)
    p[2]: Linear damping (c)
    p[3]: van der Pol damping coefficient (mu)
    p[4]: Saturation scale for cubic stiffness
    p[5]: Saturation scale for damping
    p[6]: Cross-coupling coefficient (x*v term in damping)
    p[7]: Quadratic velocity term (v^2 damping)
    p[8]: Constant force offset (bias)
    p[9]: Global scale factor
    
    To ensure numerical stability and help BFGS converge from all-ones start,
    we use tanh-saturated terms for the nonlinearities, which bounds the effective
    coefficients and prevents explosive gradients.
    """
    p = params
    
    # Extract parameters
    k_lin = p[0]
    k_cub = p[1]
    c_lin = p[2]
    c_vdp = p[3]
    s_k = p[4]
    s_c = p[5]
    c_cross = p[6]
    c_v2 = p[7]
    f_offset = p[8]
    scale = p[9]
    
    # Compute nonlinear stiffness: -k(x) * x
    # k(x) = k_lin + k_cub * tanh(s_k * x^2)
    # Using tanh(s_k * x^2) ensures the cubic-like behavior saturates, keeping values bounded.
    # This mimics hardening/softening springs.
    x2 = x * x
    k_eff = k_lin + k_cub * np.tanh(s_k * x2)
    
    # Compute nonlinear damping: -c(x, v) * v
    # c(x, v) = c_lin + c_vdp * tanh(s_c * x^2) + c_cross * x * tanh(s_c * v) + c_v2 * v * tanh(s_c * v)
    # The term c_vdp * tanh(s_c * x^2) mimics van der Pol negative damping at small amplitudes.
    # The term c_cross * x * ... adds a state-dependent coupling.
    # The term c_v2 * v * ... adds a quadratic velocity damping effect (bounded by tanh).
    c_eff = c_lin + c_vdp * np.tanh(s_c * x2) + c_cross * x * np.tanh(s_c * v) + c_v2 * v * np.tanh(s_c * v)
    
    # Total acceleration
    a = -k_eff * x - c_eff * v + f_offset
    
    # Apply global scale
    a = scale * a
    
    return a
# EVOLVE-BLOCK-END
