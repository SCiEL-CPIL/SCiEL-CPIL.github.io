import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Van der Pol + Duffing nonlinear oscillator acceleration model.
    
    Structure: a = f(x, v) with:
      - Duffing restoring: linear + cubic in x
      - Van der Pol damping: linear + (x^2 - mu)*v, giving amplitude-dependent damping
      - Higher-order corrections: x^2*v, v^2, x*v, x^4, v^4, x^2*v^2
    
    Parameters (10):
      p0: linear restoring (omega^2)
      p1: cubic restoring (Duffing beta)
      p2: linear damping (gamma)
      p3: Van der Pol mu (amplitude-dependent damping strength)
      p4: x^2*v nonlinear damping correction
      p5: v^2 velocity-dependent stiffness
      p6: x*v parametric coupling
      p7: x^4 quartic restoring
      p8: v^4 quartic damping
      p9: constant bias
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    
    # Duffing restoring force: -omega^2*x - beta*x^3 - x^4 term
    term_restore = p0 * x + p1 * x3 + p7 * x4
    
    # Van der Pol damping: gamma*v + mu*(x^2 - 1)*v + higher order
    # The (x^2 - 1)*v term gives energy injection at small x, dissipation at large x
    term_vdP_damping = p2 * v + p3 * (x2 - 1.0) * v + p4 * x2 * v
    
    # Velocity-dependent stiffness and higher-order damping
    term_vel_stiff = p5 * v2
    term_param_coup = p6 * x * v
    term_v4_damp = p8 * v4
    
    # Constant
    term_const = p9 * np.ones_like(x)
    
    result = term_restore + term_vdP_damping + term_vel_stiff + term_param_coup + term_v4_damp + term_const
    
    return result
