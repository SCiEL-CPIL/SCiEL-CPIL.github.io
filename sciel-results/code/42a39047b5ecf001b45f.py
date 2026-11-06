"""Fit a degree-5 polynomial closure for the unresolved tendency in two-scale Lorenz-96."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n < 6:
        return np.zeros(6)
    
    # Remove NaN/Inf values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    
    if n < 6:
        return np.zeros(6)
    
    # Fit degree-5 polynomial using polyfit
    # polyfit returns coefficients in ascending order of powers
    try:
        coeff = np.polynomial.polynomial.polyfit(x, y, 5)
    except Exception:
        # Fallback to lower degree if fitting fails
        try:
            coeff = np.zeros(6)
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            return coeff
        except Exception:
            return np.zeros(6)
    
    # Ensure all coefficients are finite
    if not np.all(np.isfinite(coeff)):
        # Try with fewer terms
        for deg in range(4, 0, -1):
            try:
                c = np.polynomial.polynomial.polyfit(x, y, deg)
                if np.all(np.isfinite(c)):
                    coeff = np.zeros(6)
                    coeff[:deg+1] = c
                    break
            except Exception:
                continue
        else:
            return np.zeros(6)
    
    # Check time
    if time.time() > deadline:
        return coeff
    
    return coeff
# EVOLVE-BLOCK-END
