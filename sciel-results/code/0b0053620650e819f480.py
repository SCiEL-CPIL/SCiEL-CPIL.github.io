"""Combined model: rational saturation + tanh hardening with temperature-dependent coefficients."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT_LIMIT = 140.0

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a hybrid model:
    - Term 1: Saturating rational function E(T)*strain / (1 + H*strain)
      Captures elastic-linear behavior at small strain, saturating at large strain.
    - Term 2: Tanh saturation S(T)*tanh(K(T)*strain)
      Captures the "knee" and plastic hardening/saturation.
    - Term 3: Temperature-dependent offset B(T)
    
    Parameters (10 constants):
      p0, p1, p2: Coefficients for E(T) = p0 + p1*temp + p2*temp^2
      p3: Hardening parameter H (constant)
      p4, p5: Coefficients for S(T) = p4 + p5*temp
      p6, p7: Coefficients for K(T) = p6 + p7*temp
      p8, p9: Coefficients for B(T) = p8 + p9*temp
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIMEOUT_LIMIT:
        return np.zeros_like(strain, dtype=np.float64)

    # Ensure inputs are 1D float64 arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute temperature-dependent coefficients
    # Quadratic dependence for Elastic Modulus to capture softening
    E_T = p0 + p1 * temp + p2 * temp**2
    
    # Constant Hardening modulus (saturation strain parameter)
    H = p3
    
    # Linear dependence for Saturation stress and Rate
    S_T = p4 + p5 * temp
    K_T = p6 + p7 * temp
    
    # Linear dependence for Offset
    B_T = p8 + p9 * temp
    
    # Term 1: Saturating rational function
    # For small strain, approx E_T * strain (Elastic)
    # For large strain, approx E_T / H (Saturation level)
    den = 1.0 + H * strain
    
    # Avoid division by zero or near-zero denominators
    eps = 1e-8
    den_safe = np.where(np.abs(den) < eps, np.sign(den + 1e-300) * eps, den)
    term1 = (E_T * strain) / den_safe
    
    # Term 2: Tanh saturation for hardening
    # tanh is bounded [-1, 1], smooth, captures the "knee" and plateau
    tanh_arg = K_T * strain
    term2 = S_T * np.tanh(tanh_arg)
    
    # Term 3: Offset
    term3 = B_T
    
    stress = term1 + term2 + term3
    
    # Ensure output is finite
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
# EVOLVE-BLOCK-END
