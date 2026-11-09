"""Root finder for 3 fixed two-dimensional systems using SciPy's least_squares (Levenberg-Marquardt/Trust Region)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares

def find_root_system(F, x0):
    """
    Find a root x of F(x) = 0 such that ||F(x)|| <= 1e-8.
    Uses scipy.optimize.least_squares which is robust for 2D systems.
    """
    x0 = np.array(x0, dtype=float)
    
    # Define the residual function for least_squares
    def residuals(x):
        try:
            res = F(x)
            res = np.asarray(res, dtype=float)
            if res.shape != (2,) or not np.all(np.isfinite(res)):
                return np.array([1e10, 1e10])
            return res
        except Exception:
            return np.array([1e10, 1e10])

    best_x = x0
    best_norm = np.linalg.norm(residuals(x0))
    
    # If already converged, return
    if best_norm <= 1e-8:
        return best_x

    # Primary attempt from x0
    # Using 'trf' (Trust Region Reflective) or 'dogbox' can be more robust than 'lm' (Levenberg-Marquardt)
    # 'lm' is faster but can fail if the initial guess is bad. 'trf' is more robust.
    # We'll try 'trf' first with a reasonable bound (infinity) to allow global search within the trust region.
    try:
        result = least_squares(
            residuals, 
            x0, 
            method='trf', 
            max_nfev=100, 
            xtol=1e-12, 
            ftol=1e-12, 
            gtol=1e-12
        )
        x_candidate = result.x
        f_candidate = residuals(x_candidate)
        norm_candidate = np.linalg.norm(f_candidate)
        
        if np.all(np.isfinite(x_candidate)) and norm_candidate < best_norm:
            best_x = x_candidate
            best_norm = norm_candidate
            
        if best_norm <= 1e-8:
            return best_x
    except Exception:
        pass

    # If primary attempt failed or didn't converge, try random restarts
    # This helps if x0 is far from the root or in a bad basin.
    rng = np.random.default_rng(42)
    
    for i in range(15):
        # Generate a random starting point
        # Scale based on x0 norm or default to 1.0
        scale = max(1.0, np.linalg.norm(x0))
        x_start = x0 + rng.uniform(-scale, scale, size=2)
        
        try:
            result = least_squares(
                residuals, 
                x_start, 
                method='trf', 
                max_nfev=50, 
                xtol=1e-12, 
                ftol=1e-12, 
                gtol=1e-12
            )
            x_candidate = result.x
            f_candidate = residuals(x_candidate)
            norm_candidate = np.linalg.norm(f_candidate)
            
            if np.all(np.isfinite(x_candidate)) and norm_candidate < best_norm:
                best_x = x_candidate
                best_norm = norm_candidate
                
            if best_norm <= 1e-8:
                return best_x
        except Exception:
            continue

    return best_x
# EVOLVE-BLOCK-END
