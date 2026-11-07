import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Bilinear Chebyshev basis.
    Uses products of Chebyshev polynomials T_i(u) * T_j(u) to form a basis
    that can capture non-polynomial-like features (e.g. |x| approximations via
    higher order terms) while remaining linear in the 10 parameters.
    """
    if len(params) == 0:
        return np.zeros_like(x)
    
    # Scale x to [-1, 1]
    u = x / 2.0
    
    # Compute Chebyshev polynomials up to degree 5
    T0 = np.ones_like(u)
    T1 = u
    T2 = 2 * u * u - 1
    T3 = 4 * u**3 - 3 * u
    T4 = 8 * u**4 - 8 * u**2 + 1
    T5 = 16 * u**5 - 20 * u**3 + 5 * u
    
    # Define 10 basis functions as products T_i * T_j
    # We select a set that spans up to degree 5 and includes interactions
    # Note: The specific choice of pairs is a heuristic to maximize expressiveness
    # within 10 parameters.
    
    phi0 = T0 * T0  # 1
    phi1 = T1 * T0  # u
    phi2 = T2 * T0  # 2u^2 - 1
    phi3 = T3 * T0  # 4u^3 - 3u
    phi4 = T4 * T0  # 8u^4 - 8u^2 + 1
    phi5 = T1 * T1  # u^2
    phi6 = T2 * T1  # 2u^3 - u
    phi7 = T3 * T1  # 4u^4 - 3u^2
    phi8 = T4 * T1  # 8u^5 - 8u^3 + u
    phi9 = T2 * T2  # 4u^4 - 4u^2 + 1
    
    # Linear combination
    res = params[0] * phi0
    if len(params) > 1:
        res += params[1] * phi1
    if len(params) > 2:
        res += params[2] * phi2
    if len(params) > 3:
        res += params[3] * phi3
    if len(params) > 4:
        res += params[4] * phi4
    if len(params) > 5:
        res += params[5] * phi5
    if len(params) > 6:
        res += params[6] * phi6
    if len(params) > 7:
        res += params[7] * phi7
    if len(params) > 8:
        res += params[8] * phi8
    if len(params) > 9:
        res += params[9] * phi9
        
    return res
