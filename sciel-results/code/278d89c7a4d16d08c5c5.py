"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing.
    Uses polynomial least-squares with the full degree-5 basis.
    """
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Filter out any non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 6:
        # Not enough data for degree-5 fit; fall back to zeros
        return np.zeros(6)

    # Fit degree-5 polynomial using numpy's polynomial module
    # polyfit returns coefficients in descending powers, so we need to reverse
    coeff_desc = np.polynomial.polynomial.polyfit(x, y, 5)
    # polyfit already returns in ascending order for polynomial module
    coeff = coeff_desc.copy()

    # Ensure all coefficients are finite
    if not np.all(np.isfinite(coeff)):
        return np.zeros(6)

    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Should not happen for a simple polyfit, but guard just in case
        pass

    return coeff
# EVOLVE-BLOCK-END
