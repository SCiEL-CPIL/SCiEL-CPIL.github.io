"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale
    forcing of a two-scale Lorenz-96 model.
    
    Returns 6 finite real polynomial coefficients in ASCENDING powers of x.
    """
    t_start = time.time()
    deadline = 110.0  # Safety margin before 120s limit

    # Convert inputs to numpy arrays and flatten
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)

    x = x[:n]
    y = y[:n]

    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size

    if n == 0:
        return np.zeros(6, dtype=np.float64)

    # Check time - if we've already spent too much time, return zeros
    if time.time() - t_start > deadline:
        return np.zeros(6, dtype=np.float64)

    # We want to fit: y ≈ c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4 + c5*x^5
    # This is a linear least squares problem in the coefficients.
    
    # However, polynomial fitting can be ill-conditioned. Let's use a few strategies:
    # 1. First, try a direct least squares fit with numpy's polynomial fitting
    # 2. If that fails or is ill-conditioned, use regularization or scaling

    # Strategy: Use numpy's polyfit which handles conditioning better
    # But we need to be careful about the degree - degree 5 might be overfitting
    
    # Let's think about this differently. The task says to minimize normalized error
    # on independent validation trajectories. We don't have validation data during training.
    # So we need to find a good generalizing polynomial.
    
    # Approach: Use cross-validation on the training data to select the best degree
    # or use regularization to prevent overfitting.
    
    # For now, let's try a full degree-5 fit with some regularization if needed.
    
    # First, let's normalize x to improve conditioning
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-10:
        x_std = 1.0
    
    x_norm = (x - x_mean) / x_std
    
    # Fit a degree-5 polynomial on normalized x
    # Using np.polynomial.polynomial.polyfit
    try:
        # Fit degree 5 polynomial
        coeff_norm = np.polynomial.polynomial.polyfit(x_norm, y, 5)
        
        # Convert back to original x scale
        # If y = sum_{i=0}^{5} c_i * ((x - x_mean)/x_std)^i
        # = sum_{i=0}^{5} c_i * x_std^{-i} * (x - x_mean)^i
        # We need to expand (x - x_mean)^i and collect powers of x
        
        # This conversion is complex. Let's instead fit directly on x.
        pass
    except Exception:
        pass

    # Simpler approach: Fit directly on x using least squares with the Vandermonde matrix
    # But use SVD-based solving for numerical stability
    
    # Build Vandermonde matrix: V[:, i] = x^i for i = 0, ..., 5
    # V has shape (n, 6)
    V = np.vander(x, N=6, increasing=True)
    
    # Solve using least squares with SVD for stability
    try:
        # Use np.linalg.lstsq which uses SVD
        coeff, residuals, rank, sv = np.linalg.lstsq(V, y, rcond=None)
        
        # Check if the solution is reasonable
        if not np.all(np.isfinite(coeff)):
            raise ValueError("Non-finite coefficients")
        
        # Check condition number
        if sv[-1] > 0 and sv[0] / sv[-1] > 1e15:
            # Ill-conditioned, try with regularization
            pass
    except Exception:
        # Fallback: use numpy's polynomial fitting
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
        except Exception:
            # Last resort: linear fit
            try:
                coeff = np.zeros(6)
                coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            except Exception:
                coeff = np.zeros(6)

    # Ensure coefficients are finite
    if not np.all(np.isfinite(coeff)):
        coeff = np.zeros(6)

    # Check time again
    if time.time() - t_start > deadline:
        return coeff

    # The above is a direct least squares fit. But the task mentions minimizing
    # normalized error on validation data. Since we don't have validation data,
    # we might want to use a lower degree to avoid overfitting.
    
    # Let's try multiple degrees and use a simple validation strategy:
    # Split training data into train/val, fit each degree, pick best on val.
    
    best_coeff = coeff
    best_val_error = np.inf

    # Shuffle and split
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    val_size = max(1, n // 5)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_val = x[val_idx]
    y_val = y[val_idx]

    for degree in range(0, 6):
        if time.time() - t_start > deadline:
            break
            
        if len(x_train) <= degree:
            continue
        
        try:
            V_train = np.vander(x_train, N=degree + 1, increasing=True)
            coeff_deg, _, _, _ = np.linalg.lstsq(V_train, y_train, rcond=None)
            
            if not np.all(np.isfinite(coeff_deg)):
                continue
            
            # Evaluate on validation
            V_val = np.vander(x_val, N=degree + 1, increasing=True)
            y_pred = V_val @ coeff_deg
            
            if not np.all(np.isfinite(y_pred)):
                continue
            
            val_mse = np.mean((y_val - y_pred) ** 2)
            
            if val_mse < best_val_error:
                best_val_error = val_mse
                # Pad coefficients to length 6
                best_coeff = np.zeros(6)
                best_coeff[:degree + 1] = coeff_deg
        except Exception:
            continue

    # Ensure final coefficients are finite
    if not np.all(np.isfinite(best_coeff)):
        best_coeff = np.zeros(6)

    return best_coeff
# EVOLVE-BLOCK-END
