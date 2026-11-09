import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Gaussian RBF and Polynomial basis.
    This approach uses Gaussian functions to model local nonlinearities and saturation,
    combined with polynomial terms for global linear and quadratic behavior.
    
    Terms:
    p0: x (linear restoring)
    p1: v (linear damping)
    p2: x^2 (quadratic restoring)
    p3: v^2 (quadratic damping)
    p4: x*v (coupling)
    p5: exp(-x^2) (even nonlinearity in x)
    p6: exp(-v^2) (even nonlinearity in v)
    p7: x*exp(-x^2) (odd nonlinearity in x, damped)
    p8: v*exp(-v^2) (odd nonlinearity in v, damped)
    p9: exp(-(x^2+v^2)) (radial nonlinearity)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute exponentials
    exp_neg_x2 = np.exp(-x**2)
    exp_neg_v2 = np.exp(-v**2)
    exp_neg_r2 = np.exp(-(x**2 + v**2))
    
    term_0 = p0 * x
    term_1 = p1 * v
    term_2 = p2 * x**2
    term_3 = p3 * v**2
    term_4 = p4 * x * v
    term_5 = p5 * exp_neg_x2
    term_6 = p6 * exp_neg_v2
    term_7 = p7 * x * exp_neg_x2
    term_8 = p8 * v * exp_neg_v2
    term_9 = p9 * exp_neg_r2
    
    return term_0 + term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8 + term_9
