import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter basis for symbolic regression on [-1,1] x [-1,1].
    Basis: 1, x, y, x^2, y^2, xy, sin(pi/2*x), sin(pi/2*y), cos(pi/2*x), cos(pi/2*y)
    
    This uses a hybrid polynomial/trigonometric basis with lower frequency trig terms.
    The sin(pi/2*x) terms are monotonic on [-1,1] and resemble tanh-like shapes.
    The cos(pi/2*x) terms provide smooth even curvature.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial terms
    term_0 = np.ones_like(x)
    term_1 = x
    term_2 = y
    term_3 = x**2
    term_4 = y**2
    term_5 = x * y
    
    # Trigonometric terms (frequency pi/2)
    # sin(pi/2 * x) is monotonic on [-1, 1], range [-1, 1]
    term_6 = np.sin(0.5 * np.pi * x)
    term_7 = np.sin(0.5 * np.pi * y)
    
    # cos(pi/2 * x) is even, range [0, 1] on [-1, 1]
    term_8 = np.cos(0.5 * np.pi * x)
    term_9 = np.cos(0.5 * np.pi * y)
    
    return (p0 * term_0 
            + p1 * term_1 
            + p2 * term_2 
            + p3 * term_3 
            + p4 * term_4 
            + p5 * term_5 
            + p6 * term_6 
            + p7 * term_7 
            + p8 * term_8 
            + p9 * term_9)
