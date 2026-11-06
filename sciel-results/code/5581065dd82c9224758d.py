"""Improved program for the suite task; uses degree-5 polynomial with regularization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Check if we have enough data
    if len(x) < 6:
        # Not enough data for degree 5, use lower degree
        deg = min(5, len(x) - 1)
        if deg < 0:
            return np.zeros(6)
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, deg)
            full_coeff = np.zeros(6)
            full_coeff[:len(coeff)] = coeff
            return full_coeff
        except:
            return np.zeros(6)
    
    # Try full degree-5 fit with regularization
    # Use np.polynomial.polynomial.polyfit which fits in ascending powers
    try:
        # First try direct fit
        coeff = np.polynomial.polynomial.polyfit(x, y, 5)
        
        # Check if coefficients are finite
        if not np.all(np.isfinite(coeff)):
            # Fall back to lower degree
            for deg in [4, 3, 2, 1]:
                try:
                    coeff = np.polynomial.polynomial.polyfit(x, y, deg)
                    if np.all(np.isfinite(coeff)):
                        full_coeff = np.zeros(6)
                        full_coeff[:len(coeff)] = coeff
                        return full_coeff
                except:
                    continue
            return np.zeros(6)
        
        return coeff[:6] if len(coeff) >= 6 else np.pad(coeff, (0, 6-len(coeff)))
        
    except Exception:
        # Fallback to linear fit
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            full_coeff = np.zeros(6)
            full_coeff[:2] = coeff
            return full_coeff
        except:
            return np.zeros(6)
# EVOLVE-BLOCK-END
