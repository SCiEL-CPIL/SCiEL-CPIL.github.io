import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator with Duffing, Van der Pol, and sinusoidal driving.
    Uses scaled basis functions to improve condition number for BFGS.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scaling constants based on training ranges
    Sx = 0.21
    Sv = 0.36
    
    # Basis functions
    # 1. Constant
    b0 = np.ones_like(t)
    
    # 2. Linear x
    b1 = x
    
    # 3. Nonlinear x (Duffing) - scaled
    b2 = (x ** 3) / (Sx ** 3)
    
    # 4. Linear v
    b3 = v
    
    # 5. Nonlinear v - scaled
    b4 = (v ** 3) / (Sv ** 3)
    
    # 6. Cross term x*v - scaled
    b5 = (x * v) / (Sx * Sv)
    
    # 7. Driving sin(t)
    b6 = np.sin(t)
    
    # 8. Driving cos(t)
    b7 = np.cos(t)
    
    # 9. Harmonic sin(2t)
    b8 = np.sin(2 * t)
    
    # 10. Harmonic cos(2t)
    b9 = np.cos(2 * t)
    
    # Combine
    return (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + 
            p4 * b4 + p5 * b5 + p6 * b6 + p7 * b7 + 
            p8 * b8 + p9 * b9)
