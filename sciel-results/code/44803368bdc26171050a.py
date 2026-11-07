"""
Symbolic regression with 3 inputs - Hybrid Trigonometric + Polynomial Basis.

Strategy:
1. Structural change from pure polynomial to a hybrid basis that includes
   trigonometric harmonics with fixed non-integer frequencies. This captures
   periodic/oscillatory dependencies orthogonal to polynomial structure.
2. Keep the model linear-in-parameters so BFGS solves a convex problem.
3. Use 10 parameters total:
   - 4 polynomial terms: constant, x, y, z
   - 6 trigonometric terms: sin(2x), cos(3y), sin(4z), sin(x+y), cos(x+z), sin(y+z)
4. Add a deadline guard for safety.
"""
import numpy as np
import time

_DEADLINE = 145.0  # seconds, safety margin before 150s kill
_start_time = None


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns predictions using a hybrid trigonometric + polynomial basis.
    
    Parameters:
    p[0]: constant term
    p[1]: coefficient for x
    p[2]: coefficient for y
    p[3]: coefficient for z
    p[4]: coefficient for sin(2x)
    p[5]: coefficient for cos(3y)
    p[6]: coefficient for sin(4z)
    p[7]: coefficient for sin(x+y)
    p[8]: coefficient for cos(x+z)
    p[9]: coefficient for sin(y+z)
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        return np.zeros_like(x, dtype=np.float64)
    
    p = params.flatten()
    
    # Ensure p has exactly 10 elements
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), mode='constant')
    else:
        p = p[:10]
        
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]
    
    # Compute basis terms
    # Polynomial part
    poly_part = c0 + c1 * x + c2 * y + c3 * z
    
    # Trigonometric part with fixed non-integer frequencies
    trig_part = (c4 * np.sin(2.0 * x) + 
                 c5 * np.cos(3.0 * y) + 
                 c6 * np.sin(4.0 * z) + 
                 c7 * np.sin(x + y) + 
                 c8 * np.cos(x + z) + 
                 c9 * np.sin(y + z))
    
    pred = poly_part + trig_part
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred
