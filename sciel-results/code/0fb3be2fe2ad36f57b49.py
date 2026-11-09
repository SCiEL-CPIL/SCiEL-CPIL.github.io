"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate using a Polynomial + Fourier Series basis.
    
    This approach uses a linear combination of polynomial terms and trigonometric
    harmonics. This is effective for smooth functions that may exhibit periodic
    behavior or can be well-approximated by a Fourier series on the bounded domain [-3, 3].
    
    Model:
    y = p0 + p1*x + p2*x^2 
        + p4*cos(p3*x) + p5*sin(p3*x)
        + p6*cos(2*p3*x) + p7*sin(2*p3*x)
        + p8*cos(3*p3*x) + p9*sin(3*p3*x)
        
    Parameters:
    p0-p2: Polynomial coefficients (constant, linear, quadratic)
    p3: Fundamental frequency (omega)
    p4, p5: Amplitudes for 1st harmonic (cos, sin)
    p6, p7: Amplitudes for 2nd harmonic (cos, sin)
    p8, p9: Amplitudes for 3rd harmonic (cos, sin)
    """
    p0, p1, p2, omega, c1, s1, c2, s2, c3, s3 = params
    
    # Polynomial part
    poly = p0 + p1 * x + p2 * x**2
    
    # Fourier part
    # Calculate angles
    theta1 = omega * x
    theta2 = 2.0 * omega * x
    theta3 = 3.0 * omega * x
    
    fourier = (c1 * np.cos(theta1) + s1 * np.sin(theta1) +
               c2 * np.cos(theta2) + s2 * np.sin(theta2) +
               c3 * np.cos(theta3) + s3 * np.sin(theta3))
    
    return poly + fourier
# EVOLVE-BLOCK-END
