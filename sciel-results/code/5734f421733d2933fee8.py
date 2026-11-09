import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression for a nonlinear oscillator: a = f(x, v, t)
    
    Uses a structural basis inspired by Duffing and Van der Pol oscillators,
    including parametric excitation terms.
    
    Parameters (10 total):
    p0: bias
    p1: linear x
    p2: cubic x
    p3: linear v
    p4: x^2 * v (nonlinear damping)
    p5: v^3 (cubic damping)
    p6: x * v^2 (coupling)
    p7: x * sin(t) (parametric)
    p8: v * cos(t) (parametric)
    p9: sin(t) (forcing)
    """
    # Normalize x and v to roughly [-1, 1] range to help BFGS
    # x range approx [-0.2, 0.17], scale 0.2, center -0.015
    # v range approx [-0.34, 0.36], scale 0.35, center 0.01
    
    x_n = (x + 0.015) / 0.2
    v_n = (v - 0.01) / 0.35
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis functions
    x2 = x_n * x_n
    x3 = x2 * x_n
    v2 = v_n * v_n
    v3 = v2 * v_n
    x2v = x2 * v_n
    xv2 = x_n * v2
    
    st = np.sin(t)
    ct = np.cos(t)
    
    # Parametric terms
    x_st = x_n * st
    v_ct = v_n * ct
    
    # Linear combination
    res = (p0 + 
           p1 * x_n + 
           p2 * x3 + 
           p3 * v_n + 
           p4 * x2v + 
           p5 * v3 + 
           p6 * xv2 + 
           p7 * x_st + 
           p8 * v_ct + 
           p9 * st)
           
    return res
