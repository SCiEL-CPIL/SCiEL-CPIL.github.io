"""Symbolic regression with 1 input using a well-conditioned basis + internal refinement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Use a well-conditioned basis (Chebyshev polynomials mapped to [0.1, 6])
    with 10 parameters. The function is linear in params, so BFGS from the
    evaluator should converge reliably.
    
    We also add a small internal refinement step using the given params as
    initial values to help if BFGS from the evaluator struggles.
    
    Actually, since the model is linear in params, the optimal solution can be
    found via least squares. But we don't have access to y. So we just return
    the linear combination.
    
    Key structural change: use Chebyshev polynomials instead of RBFs for
    better conditioning and global coverage.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Map x from [0.1, 6] to [-1, 1] for Chebyshev polynomials
    # t = (2*x - (0.1 + 6)) / (6 - 0.1) = (2*x - 6.1) / 5.9
    t = (2.0 * x - 6.1) / 5.9
    
    # Compute Chebyshev polynomials T_0 through T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    n_terms = 10
    basis = np.zeros((len(x), n_terms), dtype=np.float64)
    basis[:, 0] = 1.0
    if n_terms > 1:
        basis[:, 1] = t
    for k in range(2, n_terms):
        basis[:, k] = 2.0 * t * basis[:, k - 1] - basis[:, k - 2]
    
    # Linear combination
    return basis @ params
# EVOLVE-BLOCK-END
