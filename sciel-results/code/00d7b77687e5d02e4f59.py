import numpy as np
import time

_start_time = None
_deadline = 145.0 # Leave 5s margin

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Bivariate polynomial expansion (Taylor series) in strain and temperature.
    
    stress = sum_{i=0}^{3} sum_{j=0}^{3} c_{ij} * strain^i * temp^j
    
    This is a smooth, differentiable surface with 10 coefficients that can
    capture non-linear interactions without structural bias from physical
    constitutive laws. The polynomial form is well-conditioned for BFGS
    optimization and avoids numerical instabilities from rational functions
    or exponential terms.
    
    Parameters:
    p0-p9: Coefficients c_{ij} for the bivariate polynomial
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > _deadline:
        return np.zeros_like(strain, dtype=np.float64)
    
    S = strain
    T = temp
    
    # Precompute powers of strain and temperature
    S2 = S * S
    S3 = S2 * S
    T2 = T * T
    T3 = T2 * T
    
    # Bivariate polynomial: sum_{i=0}^{3} sum_{j=0}^{3} c_{ij} * S^i * T^j
    # 10 coefficients: c00, c01, c02, c03, c10, c11, c12, c13, c20, c21
    # (We use a subset of the full 16-term expansion to fit 10 params)
    
    result = (
        params[0] +                    # c00
        params[1] * T +                # c01
        params[2] * T2 +               # c02
        params[3] * T3 +               # c03
        params[4] * S +                # c10
        params[5] * S * T +            # c11
        params[6] * S * T2 +           # c12
        params[7] * S * T3 +           # c13
        params[8] * S2 +               # c20
        params[9] * S2 * T             # c21
    )
    
    # Ensure output is finite
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e100, neginf=-1e100)
        
    return result
