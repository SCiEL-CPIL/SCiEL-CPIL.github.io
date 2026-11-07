import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model with time-dependent driving.
    Uses normalized inputs to improve numerical stability for BFGS.
    Basis: 1, x, x^3, v, v^3, xv, t, t^2, xt, vt
    """
    # Normalize inputs to roughly unit scale to help optimizer
    # t in [30, 50] -> center 40, scale 10
    # x in [-0.2, 0.17] -> scale 0.2
    # v in [-0.34, 0.36] -> scale 0.35
    t_n = (t - 40.0) / 10.0
    x_n = x / 0.2
    v_n = v / 0.35

    p = params
    
    # Basis terms
    b0 = 1.0
    b1 = x_n
    b2 = x_n**3
    b3 = v_n
    b4 = v_n**3
    b5 = x_n * v_n
    b6 = t_n
    b7 = t_n**2
    b8 = x_n * t_n
    b9 = v_n * t_n

    return (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + p[4] * b4 +
            p[5] * b5 + p[6] * b6 + p[7] * b7 + p[8] * b8 + p[9] * b9)
