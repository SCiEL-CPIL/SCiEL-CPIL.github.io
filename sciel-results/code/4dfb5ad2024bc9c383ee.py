import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Polynomial interaction basis for nonlinear oscillator.
    
    Uses centered/scaled features and a 10-term polynomial basis:
    1. bias
    2. x_centered
    3. v_centered  
    4. t_centered
    5. x_centered^2
    6. v_centered^2
    7. x_centered * v_centered
    8. t_centered * x_centered
    9. t_centered * v_centered
    10. x_centered^3
    """
    # Center and scale inputs to O(1) range
    # t: [30, 50] -> center=40, range=20
    # x: [-0.2056, 0.1749] -> center=-0.01535, range=0.3805
    # v: [-0.3391, 0.3624] -> center=0.01165, range=0.7015
    
    t_c = (t - 40.0) / 10.0       # O(1), range ~[-2, 2]
    x_c = (x + 0.01535) / 0.19025  # O(1), range ~[-1, 1]
    v_c = (v - 0.01165) / 0.35075  # O(1), range ~[-1, 1]
    
    # Polynomial basis terms
    term1 = 1.0                          # bias
    term2 = x_c                          # linear x
    term3 = v_c                          # linear v
    term4 = t_c                          # linear t
    term5 = x_c ** 2                     # quadratic x
    term6 = v_c ** 2                     # quadratic v
    term7 = x_c * v_c                    # cross x*v
    term8 = t_c * x_c                    # cross t*x
    term9 = t_c * v_c                    # cross t*v
    term10 = x_c ** 3                    # cubic x (nonlinear restoring)
    
    a, b, c, d, e, f, g, h, i, j = params
    
    result = (a * term1 + b * term2 + c * term3 + d * term4 +
              e * term5 + f * term6 + g * term7 + h * term8 +
              i * term9 + j * term10)
    
    return result
