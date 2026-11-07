import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod using a smooth hyperbolic model
    with temperature-dependent coefficients and polynomial corrections.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent coefficients
    E = p0 + p1 * temp
    K = p2 + p3 * temp
    n = 1.0 + np.abs(p4)  # Ensure n >= 1 for mathematical stability
    
    # Hyperbolic model: sigma = E * eps / (1 + (E * eps / K)^n)
    # This is smooth and captures elastic-plastic transition
    E_safe = np.where(np.abs(E) < 1e-10, np.sign(E) * 1e-10, E)
    K_safe = np.where(np.abs(K) < 1e-10, np.sign(K) * 1e-10, K)
    
    # Handle sign carefully
    ratio = E_safe * strain / K_safe
    
    # Use abs for the power term to avoid issues with negative bases
    # but preserve the sign of the linear term
    ratio_abs = np.abs(ratio)
    denom = 1.0 + np.power(ratio_abs, n)
    denom_safe = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    
    # Base stress from hyperbolic model
    stress_base = E_safe * strain / denom_safe
    
    # Add polynomial corrections for better fit
    # These capture systematic deviations
    stress = stress_base + p5 * strain**2 + p6 * temp * strain + p7 * temp**2 + p8 * strain * temp**2 + p9 * strain**3
    
    return stress
