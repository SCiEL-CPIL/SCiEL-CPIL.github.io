"""Rewritten program: odd-power polynomial closure for subgrid tendency."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s limit
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # The subgrid tendency h*c*mean(Y) is approximately an odd function of X
    # due to the symmetry of the Lorenz-96 fast dynamics.
    # Fit using only odd powers: x, x^3, x^5
    # This reduces parameters from 6 to 3, improving generalization.
    
    # Build design matrix with odd powers
    A = np.column_stack([x, x**3, x**5])
    
    # Use ridge regression for stability
    # Try a small regularization
    try:
        # Solve least squares with slight ridge for numerical stability
        lam = 1e-6
        coeff_odd = np.linalg.solve(A.T @ A + lam * np.eye(3), A.T @ y)
    except np.linalg.LinAlgError:
        coeff_odd = np.linalg.lstsq(A, y, rcond=None)[0]
    
    # Map odd coefficients to full degree-5 polynomial
    coeff = np.zeros(6)
    coeff[1] = coeff_odd[0]  # x^1
    coeff[3] = coeff_odd[1]  # x^3
    coeff[5] = coeff_odd[2]  # x^5
    
    # Verify finite
    if not np.all(np.isfinite(coeff)):
        # Fallback to linear
        coeff = np.zeros(6)
        coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
    
    return coeff
# EVOLVE-BLOCK-END
