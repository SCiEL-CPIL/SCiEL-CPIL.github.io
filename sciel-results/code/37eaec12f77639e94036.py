"""Improved program for the suite task; odd-power polynomial closure exploiting antisymmetry."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    Exploits the antisymmetry of the fast-slow coupling by fitting only odd powers.
    Returns 6 coefficients in ascending powers of x.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # Ensure inputs are float64 numpy arrays
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Handle edge case of empty or tiny training sets
    if len(x) < 3:
        return np.zeros(6)

    # Remove NaN/Inf if present
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.zeros(6)

    # Check time
    if time.time() - start_time > deadline:
        return np.zeros(6)

    # The unresolved forcing in Lorenz-96 two-scale model is an odd function of x.
    # Fit only odd powers: c1*x + c3*x^3 + c5*x^5
    # This reduces parameters from 6 to 3, improving generalization.
    
    try:
        # Build design matrix for odd powers: [x, x^3, x^5]
        V = np.column_stack([x**1, x**3, x**5])
        
        # Solve least squares
        coeffs_odd, residuals, rank, sv = np.linalg.lstsq(V, y, rcond=None)
        
        # Check if solution is reasonable
        if not np.all(np.isfinite(coeffs_odd)):
            # Fallback: simpler odd fit with just x
            c1 = np.sum(x * y) / np.sum(x**2)
            coeffs_odd = np.array([c1, 0.0, 0.0])
        
        # Map to full 6-coefficient array (ascending powers)
        result = np.zeros(6)
        result[1] = coeffs_odd[0]  # x^1
        result[3] = coeffs_odd[1]  # x^3
        result[5] = coeffs_odd[2]  # x^5
        
        # Verify the fit is reasonable; if MSE is very high, try a simpler model
        y_pred = V @ coeffs_odd
        mse = np.mean((y - y_pred) ** 2)
        
        # If the cubic/quintic terms are causing issues, fall back to linear odd
        if mse > np.var(y) * 1.5 or not np.all(np.isfinite(result)):
            # Try just linear odd: c1*x
            c1 = np.sum(x * y) / np.sum(x**2)
            result = np.zeros(6)
            result[1] = c1
        
        return result
    
    except Exception:
        # Final fallback: linear odd
        try:
            c1 = np.sum(x * y) / np.sum(x**2)
            result = np.zeros(6)
            result[1] = c1
            if not np.isfinite(c1):
                result = np.zeros(6)
            return result
        except Exception:
            return np.zeros(6)
# EVOLVE-BLOCK-END
