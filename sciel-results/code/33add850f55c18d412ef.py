import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a Fourier-spectral basis.
    
    The idea: a nonlinear oscillator's dynamics, when viewed in the (x, v) plane,
    can have quasi-periodic or oscillatory structure. A Fourier basis in scaled
    coordinates can capture this more naturally than pure polynomials.
    
    a = sum_k p_k * basis_k(x, v)
    
    where basis functions are sinusoids of different frequencies in X and V,
    plus a small constant offset.
    """
    # Scale inputs to roughly [-pi, pi] range for natural sinusoidal behavior
    # x range approx [-0.57, 0.53], v range approx [-0.18, 0.28]
    X = np.pi * x / 0.55
    V = np.pi * v / 0.25
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Fourier-spectral basis: combinations of sin/cos in X and V
    # This allows capturing oscillatory/nonlinear structure in the dynamics
    term0 = p0 * np.sin(X)
    term1 = p1 * np.cos(X)
    term2 = p2 * np.sin(V)
    term3 = p3 * np.cos(V)
    term4 = p4 * np.sin(2.0 * X)
    term5 = p5 * np.cos(2.0 * X)
    term6 = p6 * np.sin(X + V)
    term7 = p7 * np.cos(X - V)
    term8 = p8 * np.sin(2.0 * V)
    term9 = p9 * np.cos(2.0 * V)
    
    return term0 + term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
