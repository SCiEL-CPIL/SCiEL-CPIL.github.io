import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a scaled polynomial basis.
    
    Inputs are scaled to approximately [-1, 1] range for well-conditioned BFGS.
    Basis: complete polynomial up to degree 3 (10 terms).
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Scale inputs to ~[-1, 1]
    xs = x / 0.5728
    vs = v / 0.2821

    # Polynomial basis up to degree 3
    return (
        p0 * 1.0
        + p1 * xs
        + p2 * vs
        + p3 * xs * xs
        + p4 * vs * vs
        + p5 * xs * vs
        + p6 * xs * xs * xs
        + p7 * vs * vs * vs
        + p8 * xs * xs * vs
        + p9 * xs * vs * vs
    )
