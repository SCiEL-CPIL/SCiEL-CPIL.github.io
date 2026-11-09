import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model with time-dependent driving force.
    
    Model:
    a = p0 * sin(p1 * t) + p2 * x + p3 * v + p4 * x^3 + p5 * v^3 + p6 * x * v + p7 * x^2 * v + p8 * v^2 * x + p9
    
    Parameters:
    p0: Drive amplitude
    p1: Drive frequency
    p2: Linear stiffness
    p3: Linear damping
    p4: Nonlinear stiffness (Duffing term)
    p5: Nonlinear damping
    p6: Velocity-position coupling (linear)
    p7: Velocity-position coupling (quadratic in x)
    p8: Velocity-position coupling (quadratic in v)
    p9: Constant bias
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Driving force
    drive = p0 * np.sin(p1 * t)
    
    # Linear terms
    linear = p2 * x + p3 * v
    
    # Nonlinear terms
    cubic = p4 * (x ** 3) + p5 * (v ** 3)
    
    # Coupling terms
    coupling = p6 * (x * v) + p7 * (x ** 2 * v) + p8 * (v ** 2 * x)
    
    # Bias
    bias = p9
    
    return drive + linear + cubic + coupling + bias
