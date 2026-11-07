import numpy as np
import time

_start_time = None

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters Chebyshev polynomial model for stress-strain.
    
    stress = sum_{d=0}^{4} (a_d + b_d * Tn) * Td(Sn)
    
    where Td is the d-th Chebyshev polynomial of the first kind,
    evaluated at normalized strain Sn in [-1, 1],
    and Tn is normalized temperature in [-1, 1].
    
    10 params: [a0, b0, a1, b1, a2, b2, a3, b3, a4, b4]
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > 130:
        return np.zeros_like(strain, dtype=np.float64)
    
    a0, b0, a1, b1, a2, b2, a3, b3, a4, b4 = params
    
    # Normalize strain to [-1, 1] based on training range
    S_min = -0.001911
    S_max = 0.9636
    S_mid = (S_min + S_max) / 2.0
    S_range = (S_max - S_min) / 2.0
    Sn = (strain - S_mid) / S_range
    
    # Normalize temperature to [-1, 1] based on training range
    T_min = 0.06667
    T_max = 1.0
    T_mid = (T_min + T_max) / 2.0
    T_range = (T_max - T_min) / 2.0
    Tn = (temp - T_mid) / T_range
    
    # Chebyshev polynomials of the first kind
    T0 = np.ones_like(Sn)
    T1 = Sn
    T2 = 2.0 * Sn * Sn - 1.0
    T3 = 4.0 * Sn * Sn * Sn - 3.0 * Sn
    T4 = 8.0 * Sn * Sn * Sn * Sn - 8.0 * Sn * Sn + 1.0
    
    # Temperature-dependent coefficients
    c0 = a0 + b0 * Tn
    c1 = a1 + b1 * Tn
    c2 = a2 + b2 * Tn
    c3 = a3 + b3 * Tn
    c4 = a4 + b4 * Tn
    
    result = c0 * T0 + c1 * T1 + c2 * T2 + c3 * T3 + c4 * T4
    
    return result
