import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod using a linear elastic term,
    a tanh saturation term, and polynomial corrections.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Linear elastic term with temperature dependence
    term_linear = (p0 + p1 * temp + p2 * temp**2) * strain
    
    # Nonlinear saturation term using tanh
    # tanh(p5 * strain) is smooth and bounded
    term_sat = (p3 + p4 * temp) * np.tanh(p5 * strain)
    
    # Polynomial corrections
    term_quad = (p6 + p7 * temp) * strain**2
    term_cubic = p8 * strain**3
    term_temp_quad = p9 * temp * strain**2
    
    stress = term_linear + term_sat + term_quad + term_cubic + term_temp_quad
    
    return stress
