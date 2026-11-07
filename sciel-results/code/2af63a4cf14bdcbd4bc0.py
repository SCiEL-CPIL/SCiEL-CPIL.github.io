import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model (Duffing-type with nonlinear damping).
    
    Model: a = -omega^2 * x - beta * x^3 - gamma * v - delta * v^3 
             + epsilon * x*v + zeta * x^2*v + theta * v^2*x + kappa * x^2 + phi * v^2
    
    10 parameters: omega^2, beta, gamma, delta, epsilon, zeta, theta, kappa, phi, offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Linear restoring force (spring)
    term_linear_x = x
    
    # Cubic nonlinear spring (Duffing)
    term_cubic_x = x**3
    
    # Linear damping
    term_linear_v = v
    
    # Cubic nonlinear damping
    term_cubic_v = v**3
    
    # Cross coupling terms
    term_xv = x * v
    term_x2v = x**2 * v
    term_v2x = v**2 * x
    
    # Quadratic terms
    term_x2 = x**2
    term_v2 = v**2
    
    # Constant offset
    term_const = 1.0
    
    return (p0 * term_linear_x + 
            p1 * term_cubic_x + 
            p2 * term_linear_v + 
            p3 * term_cubic_v + 
            p4 * term_xv + 
            p5 * term_x2v + 
            p6 * term_v2x + 
            p7 * term_x2 + 
            p8 * term_v2 + 
            p9 * term_const)
