"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a Fourier series basis in strain with temperature-dependent coefficients.
    The strain range is [-0.001911, 0.9636], so we map it to [0, 2*pi] for the
    Fourier basis. Temperature modulates the amplitude of each harmonic term.
    
    Model:
    Let x = (eps - eps_min) / (eps_max - eps_min)  # normalized strain in [0,1]
    sigma = c0 + sum_{k=1}^{K} [a_k(T) * sin(2*pi*k*x) + b_k(T) * cos(2*pi*k*x)]
    
    With 10 params, we can afford:
    - c0 (intercept)
    - For k=1: a1, b1 (2 params)
    - For k=2: a2, b2 (2 params)
    - For k=3: a3, b3 (2 params)
    - Temperature coefficient: T scales all amplitudes by (1 + p9*T)
    
    So: sigma = c0 + (1 + p9*T) * [p1*sin(2*pi*x) + p2*cos(2*pi*x) 
                                      + p3*sin(4*pi*x) + p4*cos(4*pi*x)
                                      + p5*sin(6*pi*x) + p6*cos(6*pi*x)]
    + p7*x + p8*x^2
    
    Wait, that's 9 params. Let me add another harmonic or a temperature interaction.
    
    Actually, let's use:
    sigma = (p0 + p1*T) + (1 + p2*T) * [p3*sin(2*pi*x) + p4*cos(2*pi*x) 
            + p5*sin(4*pi*x) + p6*cos(4*pi*x) + p7*sin(6*pi*x) + p8*cos(6*pi*x)]
            + p9*x^2
            
    That's 10 params.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Normalize strain to [0, 1]
    eps_min = -0.001911
    eps_max = 0.9636
    x = (eps - eps_min) / (eps_max - eps_min)
    
    # Clamp x to [0, 1] to stay in range
    x = np.clip(x, 0.0, 1.0)
    
    # Fourier basis
    s1 = np.sin(2.0 * np.pi * x)
    c1 = np.cos(2.0 * np.pi * x)
    s2 = np.sin(4.0 * np.pi * x)
    c2 = np.cos(4.0 * np.pi * x)
    s3 = np.sin(6.0 * np.pi * x)
    c3 = np.cos(6.0 * np.pi * x)
    
    # Temperature-dependent scaling
    scale = 1.0 + p2 * T
    
    # Combine Fourier terms
    fourier_term = p3 * s1 + p4 * c1 + p5 * s2 + p6 * c2 + p7 * s3 + p8 * c3
    
    # Intercept with temperature
    intercept = p0 + p1 * T
    
    # Quadratic strain correction
    quad_term = p9 * x * x
    
    sigma = intercept + scale * fourier_term + quad_term
    
    return sigma
# EVOLVE-BLOCK-END
