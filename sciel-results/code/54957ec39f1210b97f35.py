"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a Fourier series basis.
    The basis consists of cos(k*x) and sin(k*x) for k=1..5.
    This is a linear model: f(x) = sum_{k=1}^{5} (a_k cos(kx) + b_k sin(kx))
    where params = [a_1, b_1, a_2, b_2, ..., a_5, b_5].
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    
    if n == 0:
        return np.array([], dtype=np.float64)
        
    # Construct the Fourier basis matrix
    # We use 5 harmonics, providing 10 basis functions.
    # Basis: [cos(x), sin(x), cos(2x), sin(2x), ..., cos(5x), sin(5x)]
    basis_matrix = np.empty((n, 10), dtype=np.float64)
    
    # Pre-compute sin and cos for efficiency if possible, 
    # but n is small (200), so direct computation is fast and clear.
    for k in range(1, 6):
        # k=1 -> cols 0, 1; k=2 -> cols 2, 3; etc.
        col_idx = 2 * (k - 1)
        
        # Compute cos(k*x) and sin(k*x)
        # Using np.multiply for clarity/speed
        kx = k * x
        basis_matrix[:, col_idx] = np.cos(kx)
        basis_matrix[:, col_idx + 1] = np.sin(kx)
        
    # Linear combination
    # params length must be 10
    # result = sum_i basis_matrix[:, i] * params[i]
    result = basis_matrix @ params
    
    # Ensure output is finite to avoid crashing the evaluator
    # If parameters lead to overflow (unlikely with sin/cos unless params are huge, 
    # but BFGS might explore large values initially), we clip.
    # However, sin/cos are bounded [-1, 1], so result is bounded by sum(|params|).
    # If params are huge, result might be huge. The evaluator handles non-finite.
    # We just return the linear combination.
    
    return result
# EVOLVE-BLOCK-END
