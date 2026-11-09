"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y using a combined polynomial and Fourier basis.
    
    This approach combines the structural flexibility of low-degree polynomials 
    (to capture smooth non-periodic trends) with a small Fourier series 
    (to capture periodic oscillations). The basis is well-conditioned for BFGS.
    """
    t_start = time.time()
    
    n_points = len(x)
    if n_points == 0:
        return np.array([])
        
    # Map x from [-3, 3] to t in [-1, 1] for better numerical conditioning
    t = x / 3.0
    
    # We have 10 parameters.
    # Strategy: Hybrid basis
    # - params[0]: constant
    # - params[1]: linear (t)
    # - params[2]: quadratic (t^2)
    # - params[3]: cubic (t^3)
    # - params[4]: quartic (t^4)
    # - params[5]: quintic (t^5)
    # - params[6]: sin(pi*t)
    # - params[7]: cos(pi*t)
    # - params[8]: sin(2*pi*t)
    # - params[9]: cos(2*pi*t)
    
    basis = np.zeros((10, n_points))
    
    # Polynomial terms (monomial basis, low degree)
    basis[0, :] = 1.0
    basis[1, :] = t
    basis[2, :] = t * t
    basis[3, :] = t * t * t
    basis[4, :] = t * t * t * t
    basis[5, :] = t * t * t * t * t
    
    # Trigonometric terms (Fourier basis, low frequency)
    basis[6, :] = np.sin(np.pi * t)
    basis[7, :] = np.cos(np.pi * t)
    basis[8, :] = np.sin(2.0 * np.pi * t)
    basis[9, :] = np.cos(2.0 * np.pi * t)
    
    # Check time, though this should be very fast
    if time.time() - t_start > 100:
        # Fallback to linear if something is wrong (should not happen)
        return params[0] * t + params[1]
        
    # Prediction is the linear combination of basis functions with weights params
    return np.einsum('i,ij->j', params, basis)
# EVOLVE-BLOCK-END
