import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Driven Duffing oscillator with parametric excitation
    # Basis functions:
    # 0: bias
    # 1: cos(t) - fundamental drive
    # 2: sin(t) - fundamental drive phase
    # 3: x - linear position
    # 4: x^3 - cubic nonlinearity (Duffing)
    # 5: v - linear velocity (damping)
    # 6: x*v - nonlinear damping
    # 7: x*cos(t) - parametric stiffness
    # 8: v*cos(t) - parametric damping
    # 9: x^2 - quadratic nonlinearity
    
    c1 = np.cos(t)
    s1 = np.sin(t)
    
    # State terms
    x1 = x
    x2 = x**2
    x3 = x**3
    v1 = v
    xv = x * v
    
    # Parametric excitation terms
    xc1 = x * c1
    vc1 = v * c1
    
    # Combine: 10 basis functions
    return (params[0] * 1.0 +
            params[1] * c1 +
            params[2] * s1 +
            params[3] * x1 +
            params[4] * x3 +
            params[5] * v1 +
            params[6] * xv +
            params[7] * xc1 +
            params[8] * vc1 +
            params[9] * x2)
