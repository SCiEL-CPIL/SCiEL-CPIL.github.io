"""Improved program: Regularized degree-5 polynomial fit with robustness checks."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    Returns 6 coefficients in ascending powers of x.
    """
    start_time = time.time()
    # Safety margin before the 150s kill limit
    deadline = 145.0 

    # Ensure inputs are float64 numpy arrays
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Filter non-finite values to ensure numerical stability
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    # Handle edge cases
    if x.size == 0:
        return np.zeros(6)
    
    # If very few points, fall back to a simpler fit to avoid overfitting or rank deficiency
    if x.size < 6:
        # Use least squares with a lower degree if possible, or just mean
        # For simplicity and robustness, if we have fewer than 6 points, 
        # a degree-5 fit is ill-posed. We can try a linear fit or return zeros.
        # Given the context, we expect many points. But let's be safe.
        try:
            if x.size >= 2:
                coeff = np.zeros(6)
                # Fit degree min(x.size-1, 5)
                deg = min(x.size - 1, 5)
                c = np.polynomial.polynomial.polyfit(x, y, deg)
                coeff[:len(c)] = c
                return coeff
            else:
                return np.array([np.mean(y), 0, 0, 0, 0, 0])
        except Exception:
            return np.zeros(6)

    # Check time
    if time.time() > deadline:
        return np.zeros(6)

    # Degree 5 polynomial
    N = 6
    # Construct Vandermonde matrix with increasing powers: [1, x, x^2, ..., x^5]
    V = np.vander(x, N=N, increasing=True)
    
    # Compute Normal Equations: (V^T V + lambda I) c = V^T y
    A = V.T @ V
    b = V.T @ y
    
    # Choose lambda for Ridge Regression
    # We want to balance fit quality and generalization.
    # A small lambda allows non-linearity; a larger one stabilizes high-order terms.
    # The scale of A depends on the range of x. 
    # trace(A) is a good measure of the overall energy.
    trace_A = np.trace(A)
    if trace_A == 0:
        lambda_val = 1.0
    else:
        # 1e-4 is a small regularization, allowing the fit to be close to OLS
        # but preventing extreme coefficients if the matrix is ill-conditioned.
        lambda_val = 1e-4 * trace_A / N
        
    A_reg = A + lambda_val * np.eye(N)
    
    try:
        coeff = np.linalg.solve(A_reg, b)
    except np.linalg.LinAlgError:
        # Fallback to least squares if solve fails (should be rare with regularization)
        try:
            coeff, _, _, _ = np.linalg.lstsq(V, y, rcond=None)
            # Ensure coeff has length 6
            if len(coeff) < 6:
                coeff = np.pad(coeff, (0, 6 - len(coeff)))
            else:
                coeff = coeff[:6]
        except Exception:
            # Ultimate fallback
            return np.zeros(6)

    # Ensure output is a flat array of 6 floats
    coeff = np.asarray(coeff, dtype=np.float64).flatten()
    if coeff.size != 6:
        # Should not happen, but just in case
        return np.zeros(6)

    # Safety check: return if we are close to the deadline
    if time.time() > deadline:
        return coeff

    return coeff
# EVOLVE-BLOCK-END
