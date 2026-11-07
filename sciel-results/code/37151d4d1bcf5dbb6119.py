"""Symbolic regression: equation(x, params) with 10 free params."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using 10 parameters.
    Uses Chebyshev polynomials T0-T9 on the scaled interval [-1, 1].
    This basis is near-orthogonal and well-conditioned, which stabilizes
    BFGS convergence for smooth functions.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if x.ndim == 0:
        x = x.reshape(1)
        
    # Ensure finite inputs
    x = np.where(np.isfinite(x), x, 3.05)
    p = np.where(np.isfinite(p), p, 0.0)
    
    # Scale x from [0.1, 6] to [-1, 1] for better numerical conditioning
    # Midpoint: 3.05, Half-range: 2.95
    t = (x - 3.05) / 2.95
    
    # Clip to [-1, 1] to handle any slight numerical overflow or out-of-bounds
    # (though the problem states x is in [0.1, 6])
    t = np.clip(t, -1.0, 1.0)
    
    try:
        # Compute Chebyshev polynomials T0 through T9
        # T0(t) = 1
        # T1(t) = t
        # Tn(t) = 2*t*Tn-1(t) - Tn-2(t)
        
        n_points = len(t)
        basis = np.zeros((n_points, 10), dtype=np.float64)
        
        if n_points == 0:
            return np.zeros(0, dtype=np.float64)
            
        basis[:, 0] = 1.0
        basis[:, 1] = t
        
        if n_points > 0:
            T_prev = np.ones_like(t)
            T_curr = t.copy()
            
            for n in range(2, 10):
                T_next = 2.0 * t * T_curr - T_prev
                basis[:, n] = T_next
                T_prev = T_curr
                T_curr = T_next
        
        # Linear combination of basis functions
        result = basis @ p
        
        # Check for non-finite values
        if not np.all(np.isfinite(result)):
            return np.zeros_like(x, dtype=np.float64)
            
        return result
    except Exception:
        return np.zeros_like(x, dtype=np.float64)
# EVOLVE-BLOCK-END
