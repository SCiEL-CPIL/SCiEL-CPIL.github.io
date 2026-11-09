"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).flatten()
    
    n_samples = len(x)
    
    if n_samples < 6:
        # Fallback: simple linear fit if too few samples
        if n_samples < 2:
            return np.zeros(6)
        coeff = np.zeros(6)
        coeff[:2] = np.polyfit(x, y, 1)[::-1]  # polyfit returns descending order
        return coeff
    
    # Fit full degree-5 polynomial with regularization
    # Use design matrix approach for better conditioning
    V = np.vander(x, 6, increasing=True)  # Columns: x^0, x^1, ..., x^5
    
    # Add small regularization to prevent overfitting
    lambda_reg = 1e-8 * np.eye(6)
    
    # Solve regularized least squares
    try:
        A = V.T @ V + lambda_reg
        b = V.T @ y
        coeff = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback to polyfit if singular
        coeff = np.polynomial.polynomial.polyfit(x, y, 5)
    
    # Check time
    if time.time() - start_time > deadline:
        return coeff
    
    # Verify coefficients are finite
    if not np.all(np.isfinite(coeff)):
        # Fallback to linear fit
        coeff = np.zeros(6)
        coeff[:2] = np.polyfit(x, y, 1)[::-1]
    
    return coeff
# EVOLVE-BLOCK-END
