"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis function to recover an unknown function f(x) on [-2, 2].
    
    The basis is a linear combination of:
    1. Polynomials (degree 0-4) for global trends.
    2. Trigonometric functions (sin, cos with period 2) for periodicity.
    3. Gaussian RBFs for localized features.
    
    This hybrid approach is more flexible than pure Chebyshev polynomials 
    for capturing diverse functional forms.
    """
    # Deadline guard: track elapsed time
    start_time = time.time()
    
    # Ensure x is a float64 array
    x = np.asarray(x, dtype=np.float64)
    
    # Basis functions
    # Polynomials
    b0 = np.ones_like(x)
    b1 = x
    b2 = x**2
    b3 = x**3
    b4 = x**4
    
    # Trigonometric (Period 2, i.e., frequency pi/1? No, sin(pi*x) has period 2)
    # On [-2, 2], sin(pi*x) completes 2 full cycles.
    b5 = np.sin(np.pi * x)
    b6 = np.cos(np.pi * x)
    
    # Gaussian RBFs
    # Widths chosen to be local but not too narrow to avoid ill-conditioning
    # Centers spread across the domain
    b7 = np.exp(-((x - 0.8) ** 2) / 0.2)
    b8 = np.exp(-((x + 0.8) ** 2) / 0.2)
    b9 = np.exp(-((x - 0.0) ** 2) / 0.5)
    
    # Combine with parameters
    # params is an array of length 10
    pred = (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
            
    # Check for non-finite values
    if not np.all(np.isfinite(pred)):
        return np.full_like(x, 1e30)
        
    # Check deadline (should be well within 120s/150s for this simple calculation)
    elapsed = time.time() - start_time
    if elapsed > 140: # Safety margin
         # In this linear model, we can't really "return best found" 
         # as we don't do iterative optimization inside equation.
         # But returning the current pred is the best we can do.
         pass
        
    return pred
# EVOLVE-BLOCK-END
