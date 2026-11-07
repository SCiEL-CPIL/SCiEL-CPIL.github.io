"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using 10 Legendre polynomials (shifted to [-3, 3])
    to approximate an unknown function. This basis is orthogonal and well-conditioned,
    facilitating efficient optimization by BFGS.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Map x from [-3, 3] to t in [-1, 1]
    t = x / 3.0

    # Calculate Legendre Polynomials P_0 to P_9
    # P_0(t) = 1
    f0 = np.ones_like(t)
    
    # P_1(t) = t
    f1 = t
    
    # P_2(t) = 0.5 * (3t^2 - 1)
    f2 = 0.5 * (3.0 * t**2 - 1.0)
    
    # P_3(t) = 0.5 * (5t^3 - 3t)
    f3 = 0.5 * (5.0 * t**3 - 3.0 * t)
    
    # P_4(t) = 1/8 * (35t^4 - 30t^2 + 3)
    f4 = 0.125 * (35.0 * t**4 - 30.0 * t**2 + 3.0)
    
    # P_5(t) = 1/8 * (63t^5 - 70t^3 + 15t)
    f5 = 0.125 * (63.0 * t**5 - 70.0 * t**3 + 15.0 * t)
    
    # P_6(t) = 1/16 * (231t^6 - 315t^4 + 105t^2 - 5)
    f6 = 0.0625 * (231.0 * t**6 - 315.0 * t**4 + 105.0 * t**2 - 5.0)
    
    # P_7(t) = 1/16 * (429t^7 - 693t^5 + 315t^3 - 35t)
    f7 = 0.0625 * (429.0 * t**7 - 693.0 * t**5 + 315.0 * t**3 - 35.0 * t)
    
    # P_8(t) = 1/128 * (6435t^8 - 12012t^6 + 6930t^4 - 1260t^2 + 35)
    f8 = (1.0 / 128.0) * (6435.0 * t**8 - 12012.0 * t**6 + 6930.0 * t**4 - 1260.0 * t**2 + 35.0)
    
    # P_9(t) = 1/128 * (12155t^9 - 25740t^7 + 18018t^5 - 4620t^3 + 315t)
    f9 = (1.0 / 128.0) * (12155.0 * t**9 - 25740.0 * t**7 + 18018.0 * t**5 - 4620.0 * t**3 + 315.0 * t)

    # Build feature matrix (N, 10)
    features = np.column_stack((f0, f1, f2, f3, f4, f5, f6, f7, f8, f9))

    # Linear combination
    result = features @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result

# EVOLVE-BLOCK-END
