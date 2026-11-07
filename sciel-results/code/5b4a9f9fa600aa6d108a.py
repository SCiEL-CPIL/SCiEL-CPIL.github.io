import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    c0, c1, c2, c3, c4, c5, c6, omega, c8, c9 = params
    
    # Scale x and v to O(1)
    x_s = x / 0.35
    v_s = v / 0.20
    
    # Polynomial terms in state space
    term0 = c0
    term1 = c1 * x_s
    term2 = c2 * v_s
    term3 = c3 * (x_s ** 2)
    term4 = c4 * (v_s ** 2)
    term5 = c5 * (x_s ** 3)
    term6 = c6 * (v_s ** 3)
    
    # Time-dependent terms
    # Ensure omega is not zero to avoid flat gradients? No, 0 is valid (no drive).
    # However, if omega is very large, cos/sin oscillate fast.
    # BFGS might struggle if omega is initialized poorly.
    # We can clip omega to a reasonable range if needed, but let's trust the fit first.
    # If the drive is absent, c8 and c9 will go to 0.
    
    # Use t directly.
    # If the period is e.g. 2*pi, omega=1.
    # If the period is smaller, omega > 1.
    
    drive = c8 * np.cos(omega * t) + c9 * np.sin(omega * t)
    
    result = term0 + term1 + term2 + term3 + term4 + term5 + term6 + drive
    
    return result
