import numpy as np
import time

# Deadline guard: 150s wall-clock limit, use 145s to be safe
_DEADLINE = 145.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 1 input.
    
    Strategy: Use a hybrid basis that combines:
    1. Polynomial terms (degrees 0-3)
    2. Trigonometric terms (frequencies 1-2)
    3. Gaussian RBFs (centers at -1, 1)
    
    The model is linear in params, so BFGS will find the global optimum efficiently.
    """
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Ensure params is a flat array of length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
    
    try:
        # Scale x to [-1, 1] for better numerical conditioning
        xs = x / 2.0
        
        # Build basis functions
        # Polynomial: [1, xs, xs^2, xs^3]
        p0 = np.ones_like(xs)
        p1 = xs.copy()
        p2 = xs**2
        p3 = xs**3
        
        # Trig: [sin(xs), cos(xs), sin(2xs), cos(2xs)]
        t0 = np.sin(xs)
        t1 = np.cos(xs)
        t2 = np.sin(2*xs)
        t3 = np.cos(2*xs)
        
        # RBF: [exp(-(x-1)^2), exp(-(x+1)^2)]
        # Note: using original x for RBF centers to match physical scale
        r0 = np.exp(-(x - 1.0)**2)
        r1 = np.exp(-(x + 1.0)**2)
        
        # Stack into basis matrix (n x 10)
        basis = np.column_stack([p0, p1, p2, p3, t0, t1, t2, t3, r0, r1])
        
        # Linear model: y = basis @ params
        return basis @ params
        
    except Exception:
        return np.zeros_like(x)
