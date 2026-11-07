"""Combined program: trigonometric basis with rational structure for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Combined basis using trigonometric functions with a rational structure.
    
    Parameters:
    - params[0]: intercept numerator
    - params[1]: cos(pi*x) numerator
    - params[2]: sin(pi*x) numerator
    - params[3]: cos(pi*y) numerator
    - params[4]: sin(pi*y) numerator
    - params[5]: cos(pi*x)*cos(pi*y) numerator
    - params[6]: sin(pi*x)*sin(pi*y) numerator
    - params[7]: cos(pi*(x+y)) numerator
    - params[8]: cos(pi*(x-y)) numerator
    - params[9]: denominator scale (denom = 1 + params[9] * sin(pi*x)*sin(pi*y))
    
    The rational structure f = (a0 + a1*cos(pi*x) + ... + a8*cos(pi*(x-y))) / (1 + a9*sin(pi*x)*sin(pi*y))
    allows capturing more complex interactions while keeping the denominator bounded.
    """
    start_time = time.time()
    
    try:
        # Ensure inputs are float arrays
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Compute trigonometric basis terms
        cos_px = np.cos(np.pi * x)
        sin_px = np.sin(np.pi * x)
        cos_py = np.cos(np.pi * y)
        sin_py = np.sin(np.pi * y)
        cos_pxy = np.cos(np.pi * (x + y))
        cos_pxmy = np.cos(np.pi * (x - y))
        sin_px_sin_py = sin_px * sin_py
        
        # Numerator terms
        num = (
            params[0]
            + params[1] * cos_px
            + params[2] * sin_px
            + params[3] * cos_py
            + params[4] * sin_py
            + params[5] * cos_px * cos_py
            + params[6] * sin_px * sin_py
            + params[7] * cos_pxy
            + params[8] * cos_pxmy
        )
        
        # Denominator: bounded to prevent extreme values
        # denom = 1 + params[9] * sin_px * sin_py
        # Since sin_px, sin_py in [-1, 1], denom in [1-|params[9]|, 1+|params[9]|]
        # For stability, we ensure denom stays away from zero
        denom = 1.0 + params[9] * sin_px_sin_py
        
        # Safety: if denom gets too close to zero, clip it
        # This prevents division by near-zero values
        min_denom = 0.1
        max_denom = 10.0
        denom = np.clip(denom, min_denom, max_denom)
        
        pred = num / denom
        
        # Replace any non-finite values with 0 to avoid optimizer issues
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
        # Check deadline (should be well within limits, but just in case)
        if time.time() - start_time > 140:
            return np.zeros_like(x)
            
        return pred
        
    except Exception:
        return np.zeros_like(x)
# EVOLVE-BLOCK-END
