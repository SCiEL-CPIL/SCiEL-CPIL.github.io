import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    Uses normalized inputs to improve optimizer stability.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs based on training ranges to O(1) scale
    # t: [30, 50] -> mean 40, range 20. Use (t-40)/10 for approx [-1, 1]
    # x: [-0.2056, 0.1749] -> mean ~ -0.015, range ~ 0.38. Use (x+0.015)/0.2
    # v: [-0.3391, 0.3624] -> mean ~ 0.01, range ~ 0.7. Use (v-0.01)/0.35
    
    t_norm = (t - 40.0) / 10.0
    x_norm = (x + 0.015) / 0.2
    v_norm = (v - 0.01) / 0.35
    
    # Calculate terms with normalized variables
    term0 = p0
    term1 = p1 * x_norm
    term2 = p2 * (x_norm ** 3)
    term3 = p3 * v_norm
    term4 = p4 * (v_norm ** 3)
    term5 = p5 * x_norm * v_norm
    term6 = p6 * np.sin(t)
    term7 = p7 * np.cos(t)
    term8 = p8 * t_norm
    term9 = p9 * (x_norm ** 2)
    
    result = term0 + term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
    
    return result
