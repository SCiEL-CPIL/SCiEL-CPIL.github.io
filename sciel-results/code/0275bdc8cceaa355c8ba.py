"""Symbolic regression via SINDy (Sparse Identification of Nonlinear Dynamics) with LARS-based sparse regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import lstsq

def _build_library(X, max_poly_degree=3, max_trig_freq=5):
    """
    Build a library of candidate basis functions for SINDy.
    Includes: constant, linear, polynomial, trigonometric, exponential, and interaction terms.
    
    Parameters:
    -----------
    X : ndarray of shape (n_samples, n_features)
        Input data (should be normalized to [0,1] for stability)
    max_poly_degree : int
        Maximum degree for polynomial terms
    max_trig_freq : int
        Maximum frequency for trigonometric terms
    
    Returns:
    --------
    library : ndarray of shape (n_samples, n_library)
        The library of candidate basis functions
    """
    n_samples, n_features = X.shape
    features = []
    feature_names = []
    
    # Constant term
    features.append(np.ones((n_samples, 1)))
    feature_names.append(['1'])
    
    # Linear terms
    for j in range(n_features):
        features.append(X[:, j:j+1])
        feature_names.append([f'x{j}'])
    
    # Polynomial terms (up to max_poly_degree)
    for degree in range(2, max_poly_degree + 1):
        for combo in _get_combinations_with_repetition(range(n_features), degree):
            term = np.ones((n_samples, 1))
            name_parts = []
            for j in combo:
                term *= X[:, j:j+1]
                name_parts.append(f'x{j}')
            features.append(term)
            feature_names.append([''.join(name_parts)])
    
    # Trigonometric terms: sin(k * x_j), cos(k * x_j) for k=1..max_trig_freq
    for k in range(1, max_trig_freq + 1):
        for j in range(n_features):
            features.append(np.sin(k * np.pi * X[:, j:j+1]))
            feature_names.append([f'sin({k}*x{j})'])
            features.append(np.cos(k * np.pi * X[:, j:j+1]))
            feature_names.append([f'cos({k}*x{j})'])
    
    # Exponential terms: exp(x_j), exp(-x_j)
    for j in range(n_features):
        features.append(np.exp(X[:, j:j+1]))
        feature_names.append([f'exp(x{j})'])
        features.append(np.exp(-X[:, j:j+1]))
        feature_names.append([f'exp(-x{j})'])
    
    # Interaction terms: x_j * x_k for j != k
    for j in range(n_features):
        for k in range(j+1, n_features):
            features.append(X[:, j:j+1] * X[:, k:k+1])
            feature_names.append([f'x{j}*x{k}'])
    
    # Trigonometric interactions: sin(x_j)*sin(x_k), sin(x_j)*cos(x_k), etc.
    for j in range(n_features):
        for k in range(j+1, n_features):
            features.append(np.sin(np.pi * X[:, j:j+1]) * np.sin(np.pi * X[:, k:k+1]))
            feature_names.append([f'sin(x{j})*sin(x{k})'])
            features.append(np.sin(np.pi * X[:, j:j+1]) * np.cos(np.pi * X[:, k:k+1]))
            feature_names.append([f'sin(x{j})*cos(x{k})'])
            features.append(np.cos(np.pi * X[:, j:j+1]) * np.cos(np.pi * X[:, k:k+1]))
            feature_names.append([f'cos(x{j})*cos(x{k})'])
    
    # Scale each feature to have unit variance for stable regression
    library = np.hstack(features)
    n_library = library.shape[1]
    
    # Standardize columns for numerical stability
    col_means = library.mean(axis=0)
    col_stds = library.std(axis=0)
    col_stds[col_stds < 1e-10] = 1.0
    library_std = (library - col_means) / col_stds
    
    return library_std, library, col_means, col_stds

def _get_combinations_with_repetition(elements, r):
    """Generate combinations with repetition."""
    if r == 0:
        yield ()
        return
    for i, e in enumerate(elements):
        for rest in _get_combinations_with_repetition(elements[i:], r-1):
            yield (e,) + rest

def _lars_sparse_regression(X, y, n_nonzero_max=20, max_iter=100):
    """
    Perform LARS (Least Angle Regression) for sparse regression.
    Returns coefficients and number of non-zero coefficients selected.
    """
    from scipy.sparse.linalg import lsqr
    
    n_samples, n_features = X.shape
    
    # Center y
    y_mean = y.mean()
    y_centered = y - y_mean
    
    # Initialize
    beta = np.zeros(n_features)
    active_set = []
    n_active = 0
    
    # Precompute X^T X and X^T y
    XtX = X.T @ X
    Xty = X.T @ y_centered
    
    # Normalize columns of X
    col_norms = np.sqrt(np.diag(XtX))
    col_norms[col_norms < 1e-10] = 1.0
    XtX_norm = XtX / np.outer(col_norms, col_norms)
    Xty_norm = Xty / col_norms
    
    for iteration in range(max_iter):
        if len(active_set) >= n_nonzero_max:
            break
            
        # Find the predictor with maximum correlation to residual
        residual = y_centered - X @ beta
        correlations = X.T @ residual / col_norms
        
        # Mask out active predictors
        correlations[active_set] = -np.inf
        
        # Select next predictor
        j = np.argmax(correlations)
        
        if correlations[j] < 1e-10:
            break
            
        active_set.append(j)
        n_active = len(active_set)
        
        # Solve for active set using normal equations
        X_active = X[:, active_set]
        try:
            beta_active, *_ = np.linalg.lstsq(X_active, y_centered, rcond=None)
            beta[active_set] = beta_active
        except:
            break
    
    return beta, len(active_set)

def fit(X_train, y_train):
    """
    Fits a regression model using SINDy with sparse regression.
    Builds a library of candidate basis functions and uses LARS to identify
    the sparse linear combination that best explains the data.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    n_samples, n_features = X.shape
    
    # 1. Normalize inputs to [0, 1]
    min_vals = X.min(axis=0, keepdims=True)
    max_vals = X.max(axis=0, keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals < 1e-12] = 1.0
    
    X_norm = (X - min_vals) / range_vals
    
    # 2. Split into train and validation sets for model selection
    rng = np.random.RandomState(42)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    val_size = int(0.2 * n_samples)
    val_idx = indices[:val_size]
    tr_idx = indices[val_size:]
    
    X_tr = X_norm[tr_idx]
    y_tr = y[tr_idx]
    X_val = X_norm[val_idx]
    y_val = y[val_idx]
    
    # 3. Build library on training data
    library_tr, library_tr_raw, lib_means, lib_stds = _build_library(X_tr, max_poly_degree=3, max_trig_freq=4)
    
    n_tr = len(y_tr)
    n_val = len(y_val)
    
    # 4. Try different sparsity levels
    best_val_mse = float('inf')
    best_beta = None
    best_n_nonzero = 0
    
    # Try a range of sparsity levels
    sparsity_levels = [3, 5, 8, 12, 16, 20, 25, 30, 40, 50]
    
    for n_nonzero in sparsity_levels:
        if time.time() > deadline - 15:
            break
            
        try:
            # Perform LARS regression
            beta, actual_n_nonzero = _lars_sparse_regression(library_tr, y_tr, n_nonzero_max=n_nonzero, max_iter=50)
            
            # Predict on validation set
            library_val, _, _, _ = _build_library(X_val, max_poly_degree=3, max_trig_freq=4)
            y_val_pred = library_val @ beta
            
            mse_val = np.mean((y_val_pred - y_val)**2)
            
            if mse_val < best_val_mse:
                best_val_mse = mse_val
                best_beta = beta
                best_n_nonzero = actual_n_nonzero
                
        except Exception as e:
            continue
    
    # 5. If SINDy didn't find a good model, try ridge regression on the library as fallback
    if best_beta is None or best_val_mse > np.var(y_val):
        # Use ridge regression
        library_tr, library_tr_raw, lib_means, lib_stds = _build_library(X_tr, max_poly_degree=3, max_trig_freq=4)
        
        best_ridge_mse = float('inf')
        best_ridge_beta = None
        
        alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        
        for alpha in alphas:
            if time.time() > deadline - 10:
                break
                
            try:
                A = library_tr.T @ library_tr + alpha * np.eye(library_tr.shape[1])
                b = library_tr.T @ y_tr
                beta_ridge = np.linalg.solve(A, b)
                
                library_val, _, _, _ = _build_library(X_val, max_poly_degree=3, max_trig_freq=4)
                y_val_pred = library_val @ beta_ridge
                mse_val = np.mean((y_val_pred - y_val)**2)
                
                if mse_val < best_ridge_mse:
                    best_ridge_mse = mse_val
                    best_ridge_beta = beta_ridge
                    
            except:
                continue
        
        if best_ridge_beta is not None and best_ridge_mse < best_val_mse:
            best_beta = best_ridge_beta
            best_val_mse = best_ridge_mse
    
    # 6. If still no good model, use simple linear regression as ultimate fallback
    if best_beta is None:
        # Simple linear model
        X_tr_aug = np.column_stack([np.ones(n_tr), X_tr])
        try:
            beta_simple, *_ = np.linalg.lstsq(X_tr_aug, y_tr, rcond=None)
            best_beta = beta_simple
            # Mark that we're using simple linear model
            use_simple_model = True
        except:
            # Constant prediction
            y_mean = y.mean()
            def predict(X):
                X = np.asarray(X, dtype=np.float64)
                return np.full(X.shape[0], y_mean)
            return predict
    
    # 7. Refit on ALL data with the best configuration
    if 'use_simple_model' not in locals():
        use_simple_model = False
    
    if not use_simple_model:
        library_full, library_full_raw, lib_means, lib_stds = _build_library(X_norm, max_poly_degree=3, max_trig_freq=4)
        
        # Use the same sparsity level for refit
        beta_full, _ = _lars_sparse_regression(library_full, y, n_nonzero_max=best_n_nonzero, max_iter=50)
        final_beta = beta_full
        final_lib_means = lib_means
        final_lib_stds = lib_stds
    else:
        X_full_aug = np.column_stack([np.ones(n_samples), X_norm])
        final_beta, *_ = np.linalg.lstsq(X_full_aug, y, rcond=None)
        final_lib_means = None
        final_lib_stds = None
    
    # Capture variables for closure
    final_min_vals = min_vals
    final_range_vals = range_vals
    
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        X_norm_test = (X - final_min_vals) / final_range_vals
        
        if not use_simple_model:
            library_test, _, _, _ = _build_library(X_norm_test, max_poly_degree=3, max_trig_freq=4)
            return library_test @ final_beta
        else:
            X_test_aug = np.column_stack([np.ones(X_norm_test.shape[0]), X_norm_test])
            return X_test_aug @ final_beta

    return predict
# EVOLVE-BLOCK-END
