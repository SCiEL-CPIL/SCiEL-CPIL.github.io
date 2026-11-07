import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for an aluminium rod using a rational function
    with quadratic temperature dependence.
    
    sigma = N(strain, temp) / D(strain, temp)
    
    N = strain * [ (p0 + p1*T + p2*T^2) + (p3 + p4*T)*strain + (p5 + p6*T)*strain^2 ]
    D = p7 + (p8 + p9*T)*strain + p10*strain^2  -- wait, only 10 params
    
    Revised: use 10 params with free denominator intercept.
    N = strain * [ (p0 + p1*T + p2*T^2) + (p3 + p4*T)*strain + (p5 + p6*T)*strain^2 ]
    D = 1 + (p7 + p8*T)*strain + p9*strain^2 + p10... no, only 10.
    
    Keep 10 params:
    p0-p6: numerator coefficients (as before)
    p7, p8: denominator linear in strain (with T dependence)
    p9: denominator quadratic in strain
    
    The key fix: NO uniform scaling. Let BFGS work with raw values.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    u = strain
    T = temp
    
    # Temperature-dependent coefficients
    a0 = p0 + p1 * T + p2 * T * T
    a1 = p3 + p4 * T
    a2 = p5 + p6 * T
    
    num = u * (a0 + a1 * u + a2 * u * u)
    
    # Denominator: controls transition and saturation
    b1 = p7 + p8 * T
    b2 = p9
    
    den = 1.0 + b1 * u + b2 * u * u
    
    # Safe division with larger epsilon to avoid numerical issues
    sigma = num / np.where(np.abs(den) < 1e-8, 1e-8, den)
    
    return sigma
