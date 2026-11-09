import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Trigonometric and Saturation basis.
    This approach uses sine/cosine terms to capture oscillatory dynamics and periodicity,
    and tanh terms to model saturation and non-linear damping/restoring forces.
    
    Terms:
    p0: sin(x) (primary restoring, linear approx for small x)
    p1: cos(x) (even nonlinearity / offset)
    p2: sin(v) (velocity dependent restoring)
    p3: cos(v) (velocity constant / even velocity nonlinearity)
    p4: sin(x)*sin(v) (odd-odd coupling)
    p5: cos(x)*cos(v) (even-even coupling)
    p6: sin(x)*cos(v) (odd-even coupling)
    p7: cos(x)*sin(v) (even-odd coupling)
    p8: tanh(x) (saturated non-linear restoring)
    p9: tanh(v) (saturated non-linear damping)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute trigonometric and saturation functions
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_v = np.sin(v)
    cos_v = np.cos(v)
    tanh_x = np.tanh(x)
    tanh_v = np.tanh(v)
    
    # Basis functions
    term_0 = p0 * sin_x
    term_1 = p1 * cos_x
    term_2 = p2 * sin_v
    term_3 = p3 * cos_v
    term_4 = p4 * sin_x * sin_v
    term_5 = p5 * cos_x * cos_v
    term_6 = p6 * sin_x * cos_v
    term_7 = p7 * cos_x * sin_v
    term_8 = p8 * tanh_x
    term_9 = p9 * tanh_v
    
    return term_0 + term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8 + term_9
