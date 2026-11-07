import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Damped Driven Duffing Oscillator Model.
    
    Parameters:
    p[0]: Linear damping coeff (multiplies v)
    p[1]: Linear stiffness coeff (multiplies x)
    p[2]: Cubic stiffness coeff (multiplies x^3)
    p[3]: Quadratic stiffness coeff (multiplies x^2)
    p[4]: Quadratic damping coeff (multiplies v^2)
    p[5]: Cross coupling coeff (multiplies x*v)
    p[6]: Drive amplitude for cos term
    p[7]: Drive amplitude for sin term
    p[8]: Drive frequency
    p[9]: Constant offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Nonlinear terms
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    xv = x * v
    
    # Drive terms
    # Using sin/cos with frequency p8
    # Note: t is in [30, 50], so p8*t will be in [30*p8, 50*p8]
    # If p8 is 1, this is standard oscillation.
    drive_cos = np.cos(p8 * t)
    drive_sin = np.sin(p8 * t)
    
    return (
        p0 * v
        + p1 * x
        + p2 * x3
        + p3 * x2
        + p4 * v2
        + p5 * xv
        + p6 * drive_cos
        + p7 * drive_sin
        + p9
    )
