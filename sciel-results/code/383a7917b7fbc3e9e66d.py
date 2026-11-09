# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    Uses a physics-informed structure:
    - Linear and cubic-like damping
    - State-dependent restoring force (Duffing-like)
    - Coupling and asymmetry terms
    """
    p = params
    
    # Scale inputs for better conditioning
    # x range ~ [-0.57, 0.53] -> scale by 2.0
    # v range ~ [-0.18, 0.28] -> scale by 5.0
    X = x * 2.0
    V = v * 5.0
    
    # 1. Damping Terms
    # Linear damping
    damp_lin = V
    # Cubic-like damping (saturated)
    damp_cub = V * np.tanh(V * V)
    
    # 2. Restoring Force Term (Directive Core)
    # Structure: x * (p1 + p2 * tanh(p3 * x^2 + p4 * v))
    # This captures linear stiffness, cubic stiffness, and velocity-dependent stiffness
    rest_core = X * (p[1] + p[2] * np.tanh(p[3] * X * X + p[4] * V))
    
    # 3. Coupling and Asymmetry Terms
    term_xv = X * V
    term_x2 = X * X
    term_v2 = V * V
    bias = 1.0
    
    # Combine
    # p[0]: Linear Damping
    # p[1..4]: Restoring Core
    # p[5]: Cubic Damping
    # p[6]: Coupling (xv)
    # p[7]: Asymmetry (x^2)
    # p[8]: Asymmetry (v^2)
    # p[9]: Bias
    
    y = (p[0] * damp_lin - rest_core + 
         p[5] * damp_cub + 
         p[6] * term_xv + 
         p[7] * term_x2 + 
         p[8] * term_v2 + 
         p[9] * bias)
         
    return y
# EVOLVE-BLOCK-END
