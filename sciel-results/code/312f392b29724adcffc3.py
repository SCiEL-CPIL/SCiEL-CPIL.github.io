"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a Fourier series basis in strain with temperature-dependent coefficients.
    The Fourier basis captures complex non-monotonic curvature and inflection points
    inherent in aluminum's elastic-plastic transition and work-hardening behavior.
    
    Model:
    sigma = c0(T) + c1(T)*sin(w*eps) + c2(T)*cos(w*eps) + c3(T)*sin(2*w*eps)
            + c4(T)*cos(2*w*eps) + c5(T)*sin(3*w*eps) + c6(T)*cos(3*w*eps)
            + c7(T)*sin(4*w*eps) + c8(T)*cos(4*w*eps) + c9(T)*sin(5*w*eps)
    
    Each coefficient ci(T) = pi * (1 + i*T) for i=0..9, where pi are the 10 params.
    The angular frequency w is fixed to capture the relevant strain scale.
    
    Parameters (10 constants):
    p0..p9: Base amplitudes for each Fourier term
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Fixed angular frequency to capture the strain range [-0.002, ~1.0]
    # This allows the Fourier terms to resolve features across the full strain range
    w = 2.0 * np.pi
    
    # Temperature-dependent coefficients: ci(T) = pi * (1 + i*T)
    # This gives each term a linear temperature dependence with a different slope
    c0 = p0 * (1.0 + 0.0 * T)
    c1 = p1 * (1.0 + 1.0 * T)
    c2 = p2 * (1.0 + 2.0 * T)
    c3 = p3 * (1.0 + 3.0 * T)
    c4 = p4 * (1.0 + 4.0 * T)
    c5 = p5 * (1.0 + 5.0 * T)
    c6 = p6 * (1.0 + 6.0 * T)
    c7 = p7 * (1.0 + 7.0 * T)
    c8 = p8 * (1.0 + 8.0 * T)
    c9 = p9 * (1.0 + 9.0 * T)
    
    # Fourier basis functions in strain
    sin1 = np.sin(w * eps)
    cos1 = np.cos(w * eps)
    sin2 = np.sin(2.0 * w * eps)
    cos2 = np.cos(2.0 * w * eps)
    sin3 = np.sin(3.0 * w * eps)
    cos3 = np.cos(3.0 * w * eps)
    sin4 = np.sin(4.0 * w * eps)
    cos4 = np.cos(4.0 * w * eps)
    sin5 = np.sin(5.0 * w * eps)
    
    sigma = (c0 + c1 * sin1 + c2 * cos1 + c3 * sin2 + c4 * cos2 +
             c5 * sin3 + c6 * cos3 + c7 * sin4 + c8 * cos4 + c9 * sin5)
    
    return sigma
# EVOLVE-BLOCK-END
