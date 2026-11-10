"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing.
    
    Strategy: Use a full degree-5 polynomial fit with careful regularization.
    The key insight is that the unresolved tendency in Lorenz-96 has a complex
    dependence on the slow state X. A global polynomial of degree 5 should be
    sufficient to capture the main features, but we need to handle the 
    multi-regime nature (F=6, 8, 10) carefully.
    
    We'll use:
    1. Full degree-5 polynomial fit with ridge regularization to avoid 
       ill-conditioning
    2. The regularization strength is chosen to balance bias and variance
    """
    start_time = time.time()
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Remove any NaN/inf values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    n = len(x)
    if n < 6:
        # Not enough data, return zeros
        return np.zeros(6)
    
    # Build the Vandermonde matrix for degree 5 polynomial
    # V[i, j] = x[i]^j
    powers = np.arange(6)
    V = x[:, None] ** powers[None, :]
    
    # Add ridge regularization to handle ill-conditioning
    # The regularization is important because high-degree polynomials 
    # can be very sensitive to small changes in x
    # We use a moderate regularization that helps with generalization
    
    # Try a few regularization strengths and pick the best via cross-validation
    # But to keep it fast, we'll use a heuristic based on the data
    
    # Scale x to [-1, 1] range for better conditioning, then transform back
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    if x_range < 1e-10:
        x_range = 1.0
    
    # Normalize x to [-1, 1]
    x_norm = 2.0 * (x - x_min) / x_range - 1.0
    
    # Build Vandermonde with normalized x
    V_norm = x_norm[:, None] ** powers[None, :]
    
    # Fit with ridge regression for several lambda values
    # We want to find a good balance - too much regularization = high bias,
    # too little = high variance and extrapolation issues
    
    # Use a simple approach: fit with a moderate regularization
    # The optimal lambda depends on the signal-to-noise ratio
    
    # Estimate noise level
    # First do an unregularized fit to estimate residuals
    try:
        coeffs_unreg, residuals, rank, sv = np.linalg.lstsq(V_norm, y, rcond=None)
        # Estimate noise variance
        dof = max(n - 6, 1)
        noise_var = np.mean((y - V_norm @ coeffs_unreg) ** 2)
    except:
        noise_var = 1.0
    
    # Try several regularization strengths
    # Lambda is scaled relative to the data variance
    y_var = np.var(y)
    if y_var < 1e-10:
        y_var = 1.0
    
    # Regularization strengths to try
    # These are in the normalized x space
    lambda_candidates = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    
    best_coeffs = None
    best_score = np.inf
    
    # Use leave-one-out or a simple train/validation split for model selection
    # To keep it fast, use a 80/20 split
    np.random.seed(42)
    perm = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx = perm[:split]
    val_idx = perm[split:]
    
    V_train = V_norm[train_idx, :]
    V_val = V_norm[val_idx, :]
    y_train = y[train_idx]
    y_val = y[val_idx]
    
    for lam in lambda_candidates:
        # Ridge regression: (V^T V + lam * I) c = V^T y
        # But we need to be careful: the regularization should be scaled
        # appropriately. Use the standard formulation.
        A = V_train.T @ V_train
        b = V_train.T @ y_train
        
        # Add regularization
        # The regularization term is lam * diag(A) to make it scale-invariant
        # Actually, standard ridge is lam * I
        A_reg = A + lam * np.eye(6)
        
        try:
            c = np.linalg.solve(A_reg, b)
        except:
            c = np.linalg.lstsq(A_reg, b, rcond=None)[0]
        
        # Evaluate on validation set
        pred_val = V_val @ c
        mse_val = np.mean((y_val - pred_val) ** 2)
        
        if mse_val < best_score:
            best_score = mse_val
            best_coeffs = c.copy()
    
    if best_coeffs is None:
        # Fallback to unregularized
        best_coeffs = coeffs_unreg
    
    # Now transform the coefficients from normalized x space to original x space
    # x_norm = 2*(x - x_min)/x_range - 1
    # So x = (x_norm + 1) * x_range/2 + x_min
    # x = a * x_norm + b where a = x_range/2, b = x_min + x_range/2
    
    a = x_range / 2.0
    b = x_min + x_range / 2.0
    
    # If f(x_norm) = sum(c_i * x_norm^i), then
    # f(x) = sum(c_i * (x/a - 1)^i) ... wait, let me redo this.
    
    # x_norm = (2*x - x_min - x_max) / x_range = 2*(x - (x_min+x_max)/2) / x_range
    # Let mu = (x_min+x_max)/2, sigma = x_range/2
    # x_norm = (x - mu) / sigma
    
    # So f(x) = sum(c_i * ((x - mu)/sigma)^i)
    #          = sum(c_i / sigma^i * (x - mu)^i)
    #          = sum(c_i / sigma^i * sum_{j=0}^{i} binom(i,j) x^j (-mu)^(i-j))
    
    mu = (x_min + x_max) / 2.0
    sigma = x_range / 2.0
    
    if abs(sigma) < 1e-15:
        sigma = 1.0
    
    # Transform coefficients
    # For each term c_i * ((x - mu)/sigma)^i, expand using binomial theorem
    # ((x - mu)/sigma)^i = sigma^(-i) * sum_{j=0}^{i} binom(i, j) x^j (-mu)^(i-j)
    
    new_coeffs = np.zeros(6)
    for i in range(6):
        if abs(best_coeffs[i]) < 1e-15:
            continue
        coeff_i = best_coeffs[i] / (sigma ** i)
        for j in range(i + 1):
            from math import comb
            binom_ij = comb(i, j)
            mu_pow = (-mu) ** (i - j)
            new_coeffs[j] += coeff_i * binom_ij * mu_pow
    
    # Verify the transformation
    # Test on a few points
    x_test = np.linspace(x_min, x_max, 20)
    x_norm_test = (x_test - mu) / sigma
    pred_orig = np.sum(new_coeffs * x_test[:, None] ** powers[None, :], axis=1)
    pred_norm = np.sum(best_coeffs * x_norm_test[:, None] ** powers[None, :], axis=1)
    
    if np.max(np.abs(pred_orig - pred_norm)) > 1e-6 * max(1.0, np.max(np.abs(pred_norm))):
        # Transformation might have issues, fall back to direct fit in original space
        # Just do a direct ridge fit in original x space
        A_orig = V.T @ V
        b_orig = V.T @ y
        # Use a small regularization
        lam_final = 1e-4 * np.mean(np.diag(A_orig))
        A_reg_orig = A_orig + lam_final * np.eye(6)
        try:
            new_coeffs = np.linalg.solve(A_reg_orig, b_orig)
        except:
            new_coeffs = np.linalg.lstsq(A_reg_orig, b_orig, rcond=None)[0]
    
    # Final check: ensure all coefficients are finite
    if not np.all(np.isfinite(new_coeffs)):
        new_coeffs = np.zeros(6)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 100:  # Safety margin
        pass  # We're done, return
    
    return new_coeffs
# EVOLVE-BLOCK-END
