import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model with amplitude-dependent damping (Van der Pol structure).
    
    Key structural feature: (x^2 - 1) * v term encodes limit cycle dynamics
    (negative damping at small amplitudes, positive at large amplitudes).
    
    Model: a = p0*x + p1*x^3 + p2*v + p3*v^3 + p4*(x^2-1)*v + p5*x*v + p6*x^2 + p7*v^2 + p8*x^5 + p9
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Restoring forces
    term_x = x
    term_x3 = x**3
    term_x5 = x**5
    
    # Damping forces
    term_v = v
    term_v3 = v**3
    
    # Amplitude-dependent damping (Van der Pol structure)
    term_vdp = (x**2 - 1.0) * v
    
    # Cross coupling
    term_xv = x * v
    
    # Quadratic terms
    term_x2 = x**2
    term_v2 = v**2
    
    # Constant
    term_const = 1.0
    
    return (p0 * term_x + 
            p1 * term_x3 + 
            p2 * term_v + 
            p3 * term_v3 + 
            p4 * term_vdp + 
            p5 * term_xv + 
            p6 * term_x2 + 
            p7 * term_v2 + 
            p8 * term_x5 + 
            p9 * term_const)
