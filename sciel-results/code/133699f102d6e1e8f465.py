import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scale inputs to O(1) range for better numerical stability
    # x range approx [-0.2, 0.2], v range approx [-0.35, 0.35]
    xs = x * 5.0
    vs = v * 3.0
    
    # Basis functions
    # 1. Constant
    b0 = 1.0
    
    # 2. Linear position
    b1 = xs
    
    # 3. Cubic position (Duffing term)
    b2 = xs**3
    
    # 4. Sinusoidal position (softening/hardening)
    b3 = np.sin(xs)
    
    # 5. Linear velocity (damping)
    b4 = vs
    
    # 6. Cubic velocity (nonlinear damping)
    b5 = vs**3
    
    # 7. Cosine drive
    b6 = np.cos(t)
    
    # 8. Sine drive
    b7 = np.sin(t)
    
    # 9. Interaction term
    b8 = xs * vs
    
    # 10. Parametric drive
    b9 = xs * np.cos(t)
    
    # Linear combination
    result = (p0 * b0 + 
              p1 * b1 + 
              p2 * b2 + 
              p3 * b3 + 
              p4 * b4 + 
              p5 * b5 + 
              p6 * b6 + 
              p7 * b7 + 
              p8 * b8 + 
              p9 * b9)
              
    return result
