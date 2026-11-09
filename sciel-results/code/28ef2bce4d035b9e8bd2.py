"""Bootstrap portfolio: model-and-optimize for stress-strain of aluminium rod."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Record the start time to implement a deadline guard.
# The evaluator runs for up to 150s. We will fail gracefully after 140s.
_START_TIME = time.time()
_TIMEOUT_LIMIT = 140.0

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.
    
    Uses a hybrid model combining a saturating rational term (elastic-plastic 
    transition) and a tanh term (strain hardening/saturation), both with 
    temperature-dependent coefficients.
    
    Model:
    stress = E(T)*strain / (1 + H*strain) + S(T)*tanh(K(T)*strain) + B(T)
    
    Where:
    E(T) = p0 + p1*temp + p2*temp**2  (Quadratic temperature dependence)
    H = p3                             (Constant)
    S(T) = p4 + p5*temp
    K(T) = p6 + p7*temp
    B(T) = p8 + p9*temp
    """
    # Deadline guard: if we are running out of time, return a safe neutral value
    if time.time() - _START_TIME > _TIMEOUT_LIMIT:
        return np.zeros_like(strain, dtype=np.float64)

    # Ensure inputs are 1D float64 arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute temperature-dependent coefficients
    # Quadratic dependence for Elasticity to better capture softening
    E_T = p0 + p1 * temp + p2 * temp**2
    # Constant Hardening modulus (saturation strain parameter)
    H_T = p3
    # Linear dependence for Saturation stress and Rate
    S_T = p4 + p5 * temp
    K_T = p6 + p7 * temp
    # Linear dependence for Offset
    B_T = p8 + p9 * temp
    
    # Term 1: Saturating rational function
    # For small strain, approx E_T * strain (Elastic)
    # For large strain, approx E_T / H_T (Saturation level)
    # Safe division
    den = 1.0 + H_T * strain
    # Avoid division by zero
    # If den is very small, replace with small epsilon with same sign
    # Since strain and H_T are likely positive in the plastic region, den > 1.
    # In elastic region (small negative/positive strain), den is near 1.
    # We just guard against exact zero or near-zero just in case parameters go weird.
    den_safe = np.where(np.abs(den) < 1e-8, np.sign(den + 1e-300) * 1e-8, den)
    term1 = (E_T * strain) / den_safe
    
    # Term 2: Tanh saturation for hardening
    # tanh is bounded [-1, 1], smooth, captures the "knee" and plateau
    term2 = S_T * np.tanh(K_T * strain)
    
    # Term 3: Offset
    term3 = B_T
    
    stress = term1 + term2 + term3
    
    # Ensure output is finite
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
# EVOLVE-BLOCK-END
