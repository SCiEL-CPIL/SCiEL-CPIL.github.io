"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing
    of a two-scale Lorenz-96 model.
    
    Args:
        training_x: Flattened array of slow-state values
        training_subgrid_tendency: Flattened array of unresolved tendencies
    
    Returns:
        Array of 6 finite real polynomial coefficients in ascending powers of x
    """
    start_time = time.time()
    DEADLINE = 115.0  # seconds, with safety margin before 120s limit
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Remove any NaN/Inf values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 6:
        # Not enough data for degree-5 fit, return zeros
        return np.zeros(6)
    
    # Try full degree-5 polynomial fit with proper scaling
    # Scale x to [-1, 1] for better conditioning
    x_min, x_max = np.min(x), np.max(x)
    x_range = x_max - x_min
    if x_range > 0:
        x_scaled = (x - (x_min + x_max) / 2.0) / (x_range / 2.0)
    else:
        x_scaled = x - x_min
        x_range = 1.0
    
    try:
        # Fit in scaled space
        coeff_scaled = np.polynomial.polynomial.polyfit(x_scaled, y, 5)
        
        # Convert back to original x scale
        # If x_scaled = (x - mid) / half_range, then x = mid + x_scaled * half_range
        mid = (x_min + x_max) / 2.0
        half_range = x_range / 2.0
        
        # Convert polynomial from scaled to original coordinates
        # Using the relation: p(x) = p_scaled((x - mid) / half_range)
        # We need to expand this into powers of x
        # This is done by converting the polynomial representation
        
        # Use polynomial conversion: if q(t) = sum c_i * t^i where t = (x - mid) / half_range
        # then p(x) = q((x - mid) / half_range) = sum c_i * ((x - mid) / half_range)^i
        # = sum c_i * (1/half_range^i) * (x - mid)^i
        # Then expand (x - mid)^i using binomial theorem
        
        coeff_orig = np.zeros(6)
        for i in range(6):
            c_i = coeff_scaled[i] / (half_range ** i) if half_range != 0 else coeff_scaled[i]
            # Expand (x - mid)^i = sum_{j=0}^{i} C(i,j) * x^j * (-mid)^(i-j)
            from math import comb
            for j in range(i + 1):
                coeff_orig[j] += c_i * comb(i, j) * ((-mid) ** (i - j))
        
        # Check if coefficients are finite
        if not np.all(np.isfinite(coeff_orig)):
            # Fall back to simpler approach
            raise ValueError("Non-finite coefficients")
            
        return coeff_orig
    
    except Exception:
        # Fallback: use lower degree if degree-5 fails
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, min(5, len(x) - 1))
            if len(coeff) < 6:
                full_coeff = np.zeros(6)
                full_coeff[:len(coeff)] = coeff
                return full_coeff
            return coeff[:6]
        except Exception:
            # Last resort: return zeros
            return np.zeros(6)
# EVOLVE-BLOCK-END
