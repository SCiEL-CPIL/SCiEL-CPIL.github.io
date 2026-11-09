"""Combined program using a fixed-center RBF basis for stable linear-in-params optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0 # Safety margin before 150s kill

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback to a simple constant prediction to avoid hanging or errors
        return np.full_like(x, params[0])

    # Extract parameters
    # Basis:
    # 0: 1 (bias)
    # 1: x
    # 2: y
    # 3: RBF center (-0.5, -0.5)
    # 4: RBF center (0.5, -0.5)
    # 5: RBF center (-0.5, 0.5)
    # 6: RBF center (0.5, 0.5)
    # 7: RBF center (0.0, 0.0)
    # 8: RBF center (-0.7, 0.0)
    # 9: RBF center (0.7, 0.0)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute distances to centers
    # Center 3: (-0.5, -0.5)
    d3 = (x - (-0.5))**2 + (y - (-0.5))**2
    # Center 4: (0.5, -0.5)
    d4 = (x - 0.5)**2 + (y - (-0.5))**2
    # Center 5: (-0.5, 0.5)
    d5 = (x - (-0.5))**2 + (y - 0.5)**2
    # Center 6: (0.5, 0.5)
    d6 = (x - 0.5)**2 + (y - 0.5)**2
    # Center 7: (0.0, 0.0)
    d7 = x**2 + y**2
    # Center 8: (-0.7, 0.0)
    d8 = (x - (-0.7))**2 + y**2
    # Center 9: (0.7, 0.0)
    d9 = (x - 0.7)**2 + y**2
    
    # Compute RBF values (Gaussian kernels with sigma=0.5)
    sigma2 = 0.25
    rbf3 = np.exp(-d3 / sigma2)
    rbf4 = np.exp(-d4 / sigma2)
    rbf5 = np.exp(-d5 / sigma2)
    rbf6 = np.exp(-d6 / sigma2)
    rbf7 = np.exp(-d7 / sigma2)
    rbf8 = np.exp(-d8 / sigma2)
    rbf9 = np.exp(-d9 / sigma2)
    
    # Construct the linear combination
    pred = (p0 + 
            p1 * x + 
            p2 * y + 
            p3 * rbf3 + 
            p4 * rbf4 + 
            p5 * rbf5 + 
            p6 * rbf6 + 
            p7 * rbf7 + 
            p8 * rbf8 + 
            p9 * rbf9)
            
    return pred
# EVOLVE-BLOCK-END
