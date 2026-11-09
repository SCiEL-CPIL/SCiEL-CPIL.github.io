import numpy as np
from scipy.special import roots_laguerre
import time

def construct():
    """
    Constructs the list of 12 positive double-root locations for the Laguerre combination.
    Uses standard Laguerre roots scaled by a factor of 10 to maximize the effective bandwidth
    and minimize the upper bound on C4, based on observed improvements from scaled roots.
    """
    # Get the standard Laguerre polynomial roots for m=12
    try:
        roots = roots_laguerre(12)[0]
        # Scale the roots by 10 to improve the bound (lower is better)
        # This is based on the observation that scaling by 10 yielded a significant score improvement.
        scaled_roots = 10.0 * roots
        # Ensure they're positive and sorted
        scaled_roots = np.sort(scaled_roots)
        # Convert to list of floats
        result = [float(r) for r in scaled_roots]
        return result
    except Exception:
        # Fallback to seed-like values if Laguerre fails
        return [4.0 * (i + 1) for i in range(12)]
