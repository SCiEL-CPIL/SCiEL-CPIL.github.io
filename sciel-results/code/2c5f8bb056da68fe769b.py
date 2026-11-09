import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    Model: a = p0 + p1*v + p2*x + p3*x^3 + p4*tanh(p5*x + p6*v) + p7*tanh(p8*x^2 + p9*v^2)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    term_0 = p0
    term_1 = p1 * v
    term_2 = p2 * x
    term_3 = p3 * (x ** 3)
    
    # Nonlinear interaction term
    arg_1 = p5 * x + p6 * v
    term_4 = p4 * np.tanh(arg_1)
    
    # Quadratic-based nonlinearity
    arg_2 = p8 * (x ** 2) + p9 * (v ** 2)
    term_5 = p7 * np.tanh(arg_2)
    
    return term_0 + term_1 + term_2 + term_3 + term_4 + term_5
