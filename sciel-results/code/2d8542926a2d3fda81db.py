import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    # Deadline guard: 110s is safe for 120s limit
    deadline = 110.0
    
    x = np.asarray(training_x, dtype=float)
    y = np.asarray(training_subgrid_tendency, dtype=float)
    
    # Handle edge cases
    if x.size < 6:
        return np.zeros(6)
        
    # Remove NaNs/Infs
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if x.size < 6:
        return np.zeros(6)

    try:
        # Fit degree 5 polynomial
        # np.polynomial.polynomial.polyfit is stable
        coeff = np.polynomial.polynomial.polyfit(x, y, 5)
    except Exception:
        # Fallback to linear
        coeff = np.zeros(6)
        if x.size >= 2:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        return coeff

    if time.time() - start_time > deadline:
        return coeff

    # Sanity check: if coefficients are huge, the fit is unstable
    # In that case, fall back to a lower degree
    if np.any(np.abs(coeff) > 1e10):
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 3)
            coeff = np.concatenate([coeff, np.zeros(2)])
        except Exception:
            coeff = np.zeros(6)
            if x.size >= 2:
                coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
                
    return coeff
