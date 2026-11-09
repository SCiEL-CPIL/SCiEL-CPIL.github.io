import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    
    # Basis functions combining polynomial and trigonometric components
    # This hybrid approach aims to capture both smooth polynomial trends
    # and potential periodic behaviors in the hidden function.
    
    # Polynomial terms
    t0 = 1.0
    t1 = x
    t2 = y
    t3 = x**2
    t4 = y**2
    t5 = x * y
    
    # Trigonometric terms (scaled to have similar magnitude to polynomial terms on [-1,1])
    # sin and cos are bounded, so they are well-conditioned for optimization
    t6 = np.sin(x)
    t7 = np.sin(y)
    t8 = np.cos(x)
    t9 = np.cos(y)
    
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
