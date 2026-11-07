"""
Symbolic regression for a nonlinear oscillator using a physics-informed polynomial basis.
Strategy: Use a polynomial expansion of the inputs x and v up to order 4.
This captures common nonlinear oscillator dynamics (e.g., x^3, xv, x^2v) directly.
The equation is a linear combination of these polynomial terms, which BFGS solves
efficiently. Unlike RBFs, this basis has global support and may generalize better
to the test distribution if the underlying physics is smooth and polynomial-like.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# Deadline guard: evaluator kills at 150s, keep safety margin
_DEADLINE = time.time() + 130

def _time_left() -> float:
    return _DEADLINE - time.time()

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target using a linear combination of polynomial basis functions.
    
    Args:
        x: Position array (N,)
        v: Velocity array (N,)
        params: Array of 10 parameters corresponding to the basis functions.
        
    Returns:
        Prediction array (N,)
    """
    try:
        # Ensure inputs are 1D arrays
        x = np.asarray(x, dtype=float).reshape(-1)
        v = np.asarray(v, dtype=float).reshape(-1)
        
        # Define polynomial basis functions (10 terms)
        # We choose terms likely to appear in nonlinear oscillator equations:
        # 0: 1 (bias)
        # 1: x
        # 2: v
        # 3: x^2
        # 4: x*v
        # 5: v^2
        # 6: x^3
        # 7: x^2*v
        # 8: x*v^2
        # 9: v^3
        
        # Compute basis values
        # Note: We compute these efficiently using vectorized operations
        
        # Term 0: 1
        t0 = np.ones_like(x)
        
        # Term 1: x
        t1 = x
        
        # Term 2: v
        t2 = v
        
        # Term 3: x^2
        t3 = x * x
        
        # Term 4: x*v
        t4 = x * v
        
        # Term 5: v^2
        t5 = v * v
        
        # Term 6: x^3
        t6 = t3 * x
        
        # Term 7: x^2*v
        t7 = t3 * v
        
        # Term 8: x*v^2
        t8 = x * t5
        
        # Term 9: v^3
        t9 = t5 * v
        
        # Stack basis into a matrix (N, 10)
        # Using column_stack is efficient for small number of columns
        basis = np.column_stack((t0, t1, t2, t3, t4, t5, t6, t7, t8, t9))
        
        # Linear combination: (N, 10) @ (10,) -> (N,)
        pred = basis @ params
        
        # Ensure finite output
        if not np.all(np.isfinite(pred)):
            pred = np.where(np.isfinite(pred), pred, 0.0)
            
        return pred
        
    except Exception:
        # Fallback to simple linear if something goes wrong
        try:
            return params[0]*x + params[1]*v + params[2]
        except Exception:
            return np.zeros_like(x)

# EVOLVE-BLOCK-END
