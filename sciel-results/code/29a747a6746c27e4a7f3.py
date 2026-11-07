"""Symbolic regression: fit a hybrid Chebyshev-Gaussian basis via BFGS."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_BUDGET = 135.0  # safety margin before 150s kill


def _remaining_time() -> float:
    """Return remaining seconds before safety deadline."""
    return _TIME_BUDGET - (time.time() - _START_TIME)


def _hybrid_basis(x, n_cheb=5, n_gauss=5):
    """
    Compute a hybrid basis matrix.
    First n_cheb columns: Chebyshev polynomials T_0..T_{n_cheb-1} on x/2.
    Next n_gauss columns: Gaussian bumps at fixed centers.
    Total columns = n_cheb + n_gauss.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n_total = n_cheb + n_gauss
    basis = np.zeros((len(x), n_total))
    
    # 1. Chebyshev part
    t = x / 2.0
    if n_cheb > 0:
        basis[:, 0] = 1.0
    if n_cheb > 1:
        basis[:, 1] = t
    for k in range(2, n_cheb):
        basis[:, k] = 2.0 * t * basis[:, k - 1] - basis[:, k - 2]
        
    # 2. Gaussian part
    # Fixed centers spread across [-2, 2]
    centers = np.linspace(-1.5, 1.5, n_gauss)
    sigma = 0.5
    for i, c in enumerate(centers):
        col_idx = n_cheb + i
        basis[:, col_idx] = np.exp(-((x - c) ** 2) / (2.0 * sigma ** 2))
        
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluate the model: sum of params[i] * Basis_i(x).
    This is linear in params, so BFGS should converge well.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure params has exactly 10 elements
    if len(params) < 10:
        # Pad with zeros if needed
        params = np.concatenate([params, np.zeros(10 - len(params))])
    else:
        params = params[:10]
        
    # Compute the basis matrix
    basis = _hybrid_basis(x, 5, 5)
    
    # Compute predictions
    return basis @ params
# EVOLVE-BLOCK-END
