"""Combined program: robust degree-five polynomial closure with adaptive regularization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    Returns 6 coefficients in ascending powers of x.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # Ensure inputs are float64 numpy arrays
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Handle edge case of empty or insufficient data
    if len(x) < 6:
        deg = min(5, max(0, len(x) - 1))
        if deg < 0:
            return np.zeros(6)
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, deg)
            full_coeff = np.zeros(6)
            full_coeff[:len(coeff)] = coeff
            return full_coeff
        except:
            return np.zeros(6)

    # Remove any NaN or Inf if present
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 6:
        deg = min(5, max(0, len(x) - 1))
        if deg < 0:
            return np.zeros(6)
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, deg)
            full_coeff = np.zeros(6)
            full_coeff[:len(coeff)] = coeff
            return full_coeff
        except:
            return np.zeros(6)

    # Build Vandermonde matrix for degree 5 (ascending powers)
    V = np.vander(x, N=6, increasing=True)

    # Compute the normal equations: A c = b
    A = V.T @ V
    b = V.T @ y

    # Check condition number of A to decide on regularization
    try:
        cond_A = np.linalg.cond(A)
    except:
        cond_A = 1e20

    # If condition number is high, use ridge regression with a small lambda
    # Otherwise, use standard least squares
    if cond_A > 1e12:
        # Use ridge regression with adaptive lambda
        # Start with a small lambda and increase if needed
        for lambda_val in [1e-8, 1e-6, 1e-4, 1e-2]:
            try:
                I = np.eye(6)
                coeffs = np.linalg.solve(A + lambda_val * I, b)
                if np.all(np.isfinite(coeffs)):
                    # Check residual
                    residual = V @ coeffs - y
                    mse = np.mean(residual**2)
                    if mse < np.inf:
                        return coeffs
            except:
                continue
        # Fallback to lstsq if all ridge attempts fail
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(V, y, rcond=None)
            if np.all(np.isfinite(coeffs)):
                return coeffs
        except:
            pass
    else:
        # Standard least squares
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(V, y, rcond=None)
            if np.all(np.isfinite(coeffs)):
                return coeffs
        except:
            pass

    # Fallback: try polyfit with lower degrees
    for deg in [5, 4, 3, 2, 1]:
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, deg)
            if np.all(np.isfinite(coeff)):
                full_coeff = np.zeros(6)
                full_coeff[:len(coeff)] = coeff
                return full_coeff
        except:
            continue

    # Final fallback: linear fit
    try:
        coeff = np.polynomial.polynomial.polyfit(x, y, 1)
        full_coeff = np.zeros(6)
        full_coeff[:len(coeff)] = coeff
        return full_coeff
    except:
        return np.zeros(6)
# EVOLVE-BLOCK-END
