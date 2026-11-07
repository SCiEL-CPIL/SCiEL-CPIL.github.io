import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a polynomial basis tailored for a nonlinear oscillator.
    The basis includes terms consistent with Duffing-type dynamics:
    linear and cubic in x, linear in v, and cross terms.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute powers and cross terms
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    v2 = v * v
    xv = x * v
    x2v = x2 * v
    x3v = x3 * v
    
    # Basis terms:
    # 1, x, v, x^2, x^3, x^4, x*v, x^2*v, x^3*v, v^2
    term0 = np.ones_like(x)
    term1 = x
    term2 = v
    term3 = x2
    term4 = x3
    term5 = x4
    term6 = xv
    term7 = x2v
    term8 = x3v
    term9 = v2
    
    return p0 * term0 + p1 * term1 + p2 * term2 + p3 * term3 + p4 * term4 + p5 * term5 + p6 * term6 + p7 * term7 + p8 * term8 + p9 * term9
