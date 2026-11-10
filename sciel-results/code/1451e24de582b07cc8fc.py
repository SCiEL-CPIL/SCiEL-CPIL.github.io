"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s limit

    # Build Vandermonde matrix for degree-5 polynomial
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    n = len(x)
    deg = 5
    V = np.vander(x, deg + 1, increasing=True)  # columns: x^0, x^1, ..., x^5

    # Tikhonov regularization: penalize higher-order coefficients more
    # This helps generalization to the unseen F=10 validation regime
    # Lambda values: mild for low orders, stronger for high orders
    lambdas = np.array([0.0, 0.0, 1e-6, 1e-5, 1e-4, 1e-3])
    L = np.diag(lambdas)
    LT = L.T @ L

    # Solve regularized least squares: (V^T V + L^T L) c = V^T y
    try:
        A = V.T @ V + LT
        b = V.T @ y
        coeff = np.linalg.solve(A, b)
        if not np.all(np.isfinite(coeff)):
            raise ValueError("Non-finite coefficients")
        return coeff
    except Exception:
        pass

    # Fallback: unweighted polyfit degree 5
    try:
        coeff = np.polynomial.polynomial.polyfit(x, y, 5)
        if not np.all(np.isfinite(coeff)):
            raise ValueError("Non-finite coefficients")
        return coeff
    except Exception:
        pass

    # Fallback: degree-3
    try:
        coeff = np.zeros(6)
        coeff[:4] = np.polynomial.polynomial.polyfit(x, y, 3)
        if np.all(np.isfinite(coeff)):
            return coeff
    except Exception:
        pass

    # Final fallback: linear
    try:
        coeff = np.zeros(6)
        coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        if np.all(np.isfinite(coeff)):
            return coeff
    except Exception:
        pass

    # Absolute fallback: zeros
    return np.zeros(6)
# EVOLVE-BLOCK-END
