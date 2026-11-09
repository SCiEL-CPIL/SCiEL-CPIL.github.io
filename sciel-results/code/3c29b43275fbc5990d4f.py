import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    # Hybrid basis: Linear, Quadratic, and Trigonometric terms
    # This provides flexibility for both polynomial and periodic behaviors
    # while keeping the number of parameters at 10.
    
    term_0 = 1.0
    term_1 = x
    term_2 = y
    term_3 = x**2
    term_4 = y**2
    term_5 = x * y
    term_6 = np.sin(x)
    term_7 = np.sin(y)
    term_8 = np.cos(x)
    term_9 = np.cos(y)
    
    return (p[0] * term_0 + 
            p[1] * term_1 + 
            p[2] * term_2 + 
            p[3] * term_3 + 
            p[4] * term_4 + 
            p[5] * term_5 + 
            p[6] * term_6 + 
            p[7] * term_7 + 
            p[8] * term_8 + 
            p[9] * term_9)
