import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod using a temperature-modulated polynomial.
    
    Structure:
    - 4th-order polynomial in strain (captures elastic, yield knee, hardening)
    - Coefficients modulated by linear and quadratic temperature terms
    - 10 params: a0..a3 (strain poly), b0..b3 (T-linear mod), c0..c1 (T-quad mod)
    
    This removes piecewise structure entirely, giving BFGS a smooth landscape.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Base polynomial coefficients (at T=0 reference)
    a0 = p0
    a1 = p1
    a2 = p2
    a3 = p3
    
    # Linear temperature modulation of each coefficient
    b0 = p4
    b1 = p5
    b2 = p6
    b3 = p7
    
    # Quadratic temperature modulation (first two coefficients)
    c0 = p8
    c1 = p9
    
    # Effective coefficients with temperature dependence
    A0 = a0 + b0 * temp + c0 * temp**2
    A1 = a1 + b1 * temp + c1 * temp**2
    A2 = a2 + b2 * temp
    A3 = a3 + b3 * temp
    
    # Evaluate polynomial: A0 + A1*eps + A2*eps^2 + A3*eps^3
    eps = strain
    result = A0 + A1 * eps + A2 * eps**2 + A3 * eps**3
    
    return result
