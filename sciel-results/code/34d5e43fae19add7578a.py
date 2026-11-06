"""
Structural redesign: full degree-5 polynomial fit with ridge regularization
excluding the intercept, plus a time-bounded search over regularization strength.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from numpy.polynomial import polynomial as P


def _build_vandermonde(x, deg=5):
    """Build the Vandermonde matrix for polynomial basis [1, x, x^2, ..., x^5]."""
    n = x.shape[0]
    V = np.empty((n, deg + 1), dtype=np.float64)
    V[:, 0] = 1.0
    for k in range(1, deg + 1):
        V[:, k] = V[:, k - 1] * x
    return V


def _ridge_fit(x, y, lam, deg=5):
    """
    Ridge-regularized least squares for degree-`deg` polynomial.
    Regularization is applied to all coefficients EXCEPT the intercept (c_0).
    Returns coefficient array of length deg+1 in ascending powers.
    """
    V = _build_vandermonde(x, deg)
    n, p = V.shape
    # Regularization matrix: zero for intercept, lam for others
    R = np.zeros((p, p))
    R[1:, 1:] = lam * np.eye(p - 1)
    # Solve (V^T V + R) c = V^T y
    A = V.T @ V + R
    b = V.T @ y
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c = np.linalg.lstsq(A, b, rcond=None)[0]
    return c


def _normalized_mse_on_train(x, y, coeff):
    """Compute normalized MSE on the training set (proxy for validation)."""
    V = _build_vandermonde(x, 5)
    pred = V @ coeff
    mse = np.mean((pred - y) ** 2)
    var = np.var(y)
    if var < 1e-15:
        return 1.0
    return mse / var


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-5 polynomial to the unresolved subgrid tendency.
    Uses ridge regularization (intercept excluded from penalty) with a
    small search over regularization strength, bounded by wall-clock time.
    """
    t_start = time.time()
    deadline = t_start + 140.0  # 10s safety margin before 150s kill

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = x.shape[0]
    if n == 0:
        return np.zeros(6)

    # Handle degenerate cases
    if n < 6:
        # Not enough data for degree-5; fall back to lower degree
        deg = max(1, n - 1)
        coeff_full = np.zeros(6)
        V = _build_vandermonde(x, deg)
        try:
            c = np.linalg.lstsq(V, y, rcond=None)[0]
            coeff_full[:deg + 1] = c
        except Exception:
            coeff_full[0] = np.mean(y)
            coeff_full[1] = np.cov(x, y, ddof=0)[0, 1] / (np.var(x) + 1e-15)
        return coeff_full

    # Search over regularization strengths
    # Log-spaced grid from very small to moderate values
    lam_values = np.logspace(-8, 2, 60)

    best_coeff = None
    best_score = np.inf

    for lam in lam_values:
        if time.time() > deadline:
            break
        coeff = _ridge_fit(x, y, lam)
        # Check for NaN/Inf
        if not np.all(np.isfinite(coeff)):
            continue
        score = _normalized_mse_on_train(x, y, coeff)
        if score < best_score:
            best_score = score
            best_coeff = coeff.copy()

    # If the search didn't find anything (shouldn't happen), fall back to unregularized
    if best_coeff is None:
        V = _build_vandermonde(x, 5)
        try:
            best_coeff = np.linalg.lstsq(V, y, rcond=None)[0]
        except Exception:
            best_coeff = np.zeros(6)
            best_coeff[0] = np.mean(y)

    # Final safety: ensure all coefficients are finite
    if not np.all(np.isfinite(best_coeff)):
        best_coeff = np.nan_to_num(best_coeff, nan=0.0, posinf=0.0, neginf=0.0)

    return best_coeff
# EVOLVE-BLOCK-END
