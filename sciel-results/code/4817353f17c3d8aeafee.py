import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    
    # Basis: Polynomials, Radial, and Exponentials
    t0 = 1.0
    t1 = x
    t2 = y
    t3 = x**2
    t4 = y**2
    t5 = x * y
    t6 = x**2 + y**2
    t7 = np.exp(x**2 + y**2)
    t8 = np.exp(x)
    t9 = np.exp(y)
    
    return (p[0] * t0 + 
            p[1] * t1 + 
            p[2] * t2 + 
            p[3] * t3 + 
            p[4] * t4 + 
            p[5] * t5 + 
            p[6] * t6 + 
            p[7] * t7 + 
            p[8] * t8 + 
            p[9] * t9)
