# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Bivariate Chebyshev polynomial expansion in strain and temperature.
    
    Uses 10 fitted constants for a complete expansion of total degree 3.
    Normalizes inputs to [-1, 1] based on training ranges to improve conditioning.
    """
    e = strain
    t = temp
    p = params

    # Normalization to [-1, 1]
    # Training ranges from task description
    e_min, e_max = -0.001911, 0.9636
    t_min, t_max = 0.06667, 1.0
    
    # Avoid division by zero if range is 0 (unlikely here)
    e_range = e_max - e_min
    t_range = t_max - t_min
    
    x = (2 * e - (e_min + e_max)) / e_range
    y = (2 * t - (t_min + t_max)) / t_range
    
    # Clip to [-1, 1] to handle slight out-of-bounds in test or numerical noise
    # Chebyshev polynomials grow rapidly outside [-1, 1], which can destabilize BFGS
    x = np.clip(x, -1.0, 1.0)
    y = np.clip(y, -1.0, 1.0)
    
    # Chebyshev polynomials of the first kind
    T0x = np.ones_like(x)
    T1x = x
    T2x = 2 * x**2 - 1
    T3x = 4 * x**3 - 3 * x
    
    T0y = np.ones_like(y)
    T1y = y
    T2y = 2 * y**2 - 1
    T3y = 4 * y**3 - 3 * y
    
    # Basis functions: T_i(x) * T_j(y) for i+j <= 3
    # 1. (0,0)
    term1 = T0x * T0y
    # 2. (1,0)
    term2 = T1x * T0y
    # 3. (2,0)
    term3 = T2x * T0y
    # 4. (3,0)
    term4 = T3x * T0y
    # 5. (0,1)
    term5 = T0x * T1y
    # 6. (1,1)
    term6 = T1x * T1y
    # 7. (2,1)
    term7 = T2x * T1y
    # 8. (0,2)
    term8 = T0x * T2y
    # 9. (1,2)
    term9 = T1x * T2y
    # 10. (0,3)
    term10 = T0x * T3y
    
    return (p[0] * term1 +
            p[1] * term2 +
            p[2] * term3 +
            p[3] * term4 +
            p[4] * term5 +
            p[5] * term6 +
            p[6] * term7 +
            p[7] * term8 +
            p[8] * term9 +
            p[9] * term10)
# EVOLVE-BLOCK-END
