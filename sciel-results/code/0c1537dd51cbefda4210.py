"""Rewritten program: Tikhonov-regularized degree-5 polynomial fit with internal k-fold CV."""
import numpy as np


def _build_design_matrix(x, degree):
    """Build Vandermonde matrix in ascending powers."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    # Use np.vander with increasing=True for ascending powers
    V = np.vander(x, N=degree + 1, increasing=True)
    return V


def _ridge_solve(V, y, lam):
    """Solve ridge regression: (V^T V + lam * I) w = V^T y"""
    n_params = V.shape[1]
    A = V.T @ V
    b = V.T @ y
    # Add regularization to diagonal
    reg = lam * np.eye(n_params)
    # Do not regularize the intercept (constant term) as heavily? 
    # Standard Tikhonov regularizes all. But often intercept is not penalized.
    # Let's penalize all coefficients for simplicity, or just non-constant.
    # The task says "L2 norm penalty on the coefficients". Usually means all.
    # However, to be safe and standard, let's penalize all.
    A_reg = A + reg
    w = np.linalg.solve(A_reg, b)
    return w


def construct(training_x, training_subgrid_tendency):
    training_x = np.asarray(training_x, dtype=np.float64)
    training_subgrid_tendency = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    n = len(training_x)
    degree = 5
    
    # Handle edge cases
    if n < degree + 1:
        # Fall back to lower degree or simple mean
        if n == 0:
            return np.zeros(6)
        # Fit a polynomial of degree min(degree, n-1)
        deg_eff = min(degree, n - 1)
        try:
            coeff = np.polynomial.polynomial.polyfit(training_x, training_subgrid_tendency, deg_eff)
            # Pad to length 6
            full_coeff = np.zeros(6)
            full_coeff[:len(coeff)] = coeff
            return full_coeff
        except:
            return np.zeros(6)

    V = _build_design_matrix(training_x, degree)
    
    # Define a grid of regularization strengths to test
    # Log-spaced from very small to large
    # We need to pick a good range. 
    # Let's use a relative scale based on the data variance.
    # Or just a fixed log-space.
    # Typical lambda range: 1e-6 to 1e6
    lambdas = np.logspace(-6, 6, 25)
    
    # K-fold cross-validation
    # Choose k=5
    k = 5
    if n < k:
        k = max(2, n // 2)
        if k < 2:
            # Just use the whole data with a small lambda
            lam = 1e-3
            w = _ridge_solve(V, training_subgrid_tendency, lam)
            return w
    
    # Create indices
    indices = np.arange(n)
    # Shuffle indices for CV folds to ensure random split
    rng = np.random.default_rng(42) # Fixed seed for determinism
    rng.shuffle(indices)
    
    # Split into k folds
    fold_indices = np.array_split(indices, k)
    
    best_mse = np.inf
    best_w = np.zeros(6)
    
    for lam in lambdas:
        mse_sum = 0.0
        for i in range(k):
            val_idx = fold_indices[i]
            train_idx = np.concatenate([fold_indices[j] for j in range(k) if j != i])
            
            V_train = V[train_idx]
            y_train = training_subgrid_tendency[train_idx]
            V_val = V[val_idx]
            y_val = training_subgrid_tendency[val_idx]
            
            w_cv = _ridge_solve(V_train, y_train, lam)
            pred_val = V_val @ w_cv
            mse = np.mean((pred_val - y_val) ** 2)
            mse_sum += mse
        
        avg_mse = mse_sum / k
        
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_w = _ridge_solve(V, training_subgrid_tendency, lam)
            
    return best_w
