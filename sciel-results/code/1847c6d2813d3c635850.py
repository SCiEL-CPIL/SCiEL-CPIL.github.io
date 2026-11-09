import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Quadratic model with selected interactions for E. coli growth rate.
    
    Features:
    1. Bias
    2. b (density)
    3. s (substrate)
    4. T (temperature)
    5. pH
    6. b*s (interaction)
    7. T*pH (interaction)
    8. b^2 (quadratic)
    9. s^2 (quadratic)
    10. T^2 (quadratic)
    
    params layout (10 constants):
      p[0]: bias
      p[1]: coeff for b
      p[2]: coeff for s
      p[3]: coeff for T
      p[4]: coeff for pH
      p[5]: coeff for b*s
      p[6]: coeff for T*pH
      p[7]: coeff for b^2
      p[8]: coeff for s^2
      p[9]: coeff for T^2
    """
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Linear terms
    y = p0 + p1 * b + p2 * s + p3 * temp + p4 * pH
    
    # Interaction terms
    y += p5 * (b * s)
    y += p6 * (temp * pH)
    
    # Quadratic terms
    y += p7 * (b ** 2)
    y += p8 * (s ** 2)
    y += p9 * (temp ** 2)
    
    return y
