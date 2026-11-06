"""Improved program for the suite task; robust degree-five polynomial closure with degree selection."""
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

    # Handle edge case of empty or tiny training sets
    if len(x) < 6:
        return np.zeros(6)

    # Remove NaN/Inf if present
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 6:
        return np.zeros(6)

    # Check time
    if time.time() - start_time > deadline:
        return np.zeros(6)

    # Strategy: Try fitting polynomials of degrees 0 through 5
    # Use AIC-like criterion to select the best degree
    # Then return the coefficients padded to degree 5

    mse_by_degree = {}
    coeff_by_degree = {}

    for deg in range(6):
        if time.time() - start_time > deadline - 5.0:
            break

        try:
            # Use np.polynomial.polynomial.polyfit for numerically stable least-squares
            coeffs = np.polynomial.polynomial.polyfit(x, y, deg)

            # Compute training MSE
            V = np.vander(x, N=deg+1, increasing=True)
            y_pred = V @ coeffs
            mse = np.mean((y - y_pred) ** 2)

            mse_by_degree[deg] = mse
            coeff_by_degree[deg] = coeffs
        except Exception:
            continue

    # If we have results, pick the best degree based on AIC-like criterion
    if len(mse_by_degree) > 0:
        n = len(x)
        aic_scores = {}
        for deg, mse in mse_by_degree.items():
            if mse > 1e-20:
                aic = n * np.log(mse) + 2 * (deg + 1)
            else:
                aic = n * np.log(1e-20) + 2 * (deg + 1)
            aic_scores[deg] = aic

        best_deg = min(aic_scores, key=aic_scores.get)

        # Return coefficients padded to degree 5
        result = np.zeros(6)
        if best_deg in coeff_by_degree:
            result[:best_deg+1] = coeff_by_degree[best_deg]

        # Ensure all coefficients are finite
        if not np.all(np.isfinite(result)):
            # Fallback to linear fit
            try:
                coeffs_lin = np.polynomial.polynomial.polyfit(x, y, 1)
                result = np.zeros(6)
                result[:2] = coeffs_lin
            except Exception:
                result = np.zeros(6)

        return result
    else:
        # Fallback: try degree 1
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            result = np.zeros(6)
            result[:2] = coeff
        except Exception:
            result = np.zeros(6)

        # Ensure all coefficients are finite
        if not np.all(np.isfinite(result)):
            result = np.zeros(6)

        return result
# EVOLVE-BLOCK-END
