"""Symbolic regression using Kernel Ridge Regression with Sum-of-Sines kernel."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    """
    Fit a regression model using Kernel Ridge Regression with a Sum-of-Sines kernel.
    The kernel is designed to capture smooth, potentially periodic nonlinear functions
    on bounded domains.
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin (total limit is 330s)
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape
    
    # Normalize inputs to [0, 1] for consistent kernel scaling
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Define frequency grid for the Sum-of-Sines kernel
    # The kernel is K(x, z) = sum_i exp(-gamma_i * (x - z)^2) * cos(omega_i * (x - z))
    # or more simply, a combination of RBF and periodic components.
    # Let's use a composite kernel: K(x, z) = K_rbf(x, z) + K_sines(x, z)
    # K_rbf(x, z) = exp(-gamma * ||x-z||^2)
    # K_sines(x, z) = sum_{k=1}^{K} cos(omega_k * (x-z)) for each dimension? 
    # Actually, a common approach for bounded domains is to use a Fourier basis.
    # Let's try a kernel based on a finite Fourier series.
    # K(x, z) = sum_{k=1}^{K} cos(k * pi * (x - z)) for 1D, generalized for dD.
    # For multi-dim, we can use product kernels or sum of per-dim kernels.
    # Let's use a sum of per-dimension Fourier features as a kernel.
    # K(x, z) = sum_{j=1}^{d} sum_{k=1}^{K} cos(k * pi * (x_j - z_j))
    
    # However, pure cosine might not capture all functions.
    # Let's combine RBF and Fourier features.
    # We will construct a feature matrix explicitly for simplicity and stability.
    
    # Feature construction:
    # 1. RBF features with a few different bandwidths
    # 2. Fourier features (sin and cos) with a few frequencies
    
    # To avoid huge feature matrices, we'll use a random projection or fixed grid.
    # But with n=200, we can afford a moderate number of features.
    
    # Let's try a different approach: Use a Neural Network-like basis? No, keep it simple.
    # Let's use a combination of polynomial and Fourier features, selected via regularization.
    
    # Actually, the prompt suggests "Sum-of-Sines kernel".
    # Let's implement a kernel matrix directly and solve the KRR.
    
    # Kernel: K(x, z) = sum_{m=1}^{M} cos(w_m * (x - z))
    # This is a valid PSD kernel if we choose w_m appropriately.
    # For multi-dim, K(x, z) = sum_{j=1}^{d} sum_{m=1}^{M} cos(w_m * (x_j - z_j))
    
    # Let's define a set of frequencies.
    # Since inputs are in [1, range], and normalized to [0,1],
    # frequencies should be multiples of pi.
    
    M = 10  # Number of frequency components per dimension
    
    def compute_kernel(X1, X2, freqs):
        """
        Compute the Sum-of-Sines kernel between X1 and X2.
        X1: (n1, d), X2: (n2, d)
        freqs: list of frequencies
        """
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        d = X1.shape[1]
        
        K = np.zeros((n1, n2))
        
        for j in range(d):
            diff = X1[:, j:j+1] - X2[j+1 if False else 0:0, j:j+1] # This is wrong, need broadcasting
            # Correct broadcasting:
            diff = X1[:, j:j+1] - X2[:, j:j+1].T
            for w in freqs:
                K += np.cos(w * diff)
        
        return K
    
    # Define frequencies
    # We'll use frequencies: pi, 2pi, ..., M*pi
    freqs = [k * np.pi for k in range(1, M + 1)]
    
    # Compute training kernel
    K_train = compute_kernel(X_norm, X_norm, freqs)
    
    # Add regularization
    # KRR solution: alpha = (K + lambda * I)^-1 y
    # We need to choose lambda. Let's use a small value.
    # We can also try a few values and pick the best via cross-validation.
    
    # For speed, let's first try a fixed lambda.
    # If time permits, we can do a small CV loop.
    
    # Let's try a few lambdas and pick the best via 5-fold CV.
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    
    best_alpha = None
    best_lambda = None
    best_cv_score = np.inf
    
    # 5-Fold Cross-Validation
    n_folds = 5
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    
    for lam in lambdas:
        if time.time() > deadline - 10:
            break
        
        cv_mse = 0
        for fold in range(n_folds):
            val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
            train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
            
            X_tr = X_norm[train_idx]
            y_tr = y[train_idx]
            X_val = X_norm[val_idx]
            y_val = y[val_idx]
            
            K_tr = compute_kernel(X_tr, X_tr, freqs)
            K_val = compute_kernel(X_val, X_tr, freqs)
            
            try:
                alpha = np.linalg.solve(K_tr + lam * np.eye(len(X_tr)), y_tr)
                pred = K_val @ alpha
                mse = np.mean((pred - y_val) ** 2)
                cv_mse += mse
            except np.linalg.LinAlgError:
                cv_mse = np.inf
                break
        
        cv_mse /= n_folds
        if cv_mse < best_cv_score:
            best_cv_score = cv_mse
            best_lambda = lam
    
    # Fit final model with best lambda
    if best_lambda is None:
        best_lambda = 1e-2
        
    K_train = compute_kernel(X_norm, X_norm, freqs)
    try:
        alpha = np.linalg.solve(K_train + best_lambda * np.eye(n), y)
    except np.linalg.LinAlgError:
        # Fallback to linear regression if KRR fails
        X_b = np.hstack([X_norm, np.ones((n, 1))])
        try:
            w, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=None)
            def predict(X_test):
                X_test = np.asarray(X_test, dtype=float)
                X_test_norm = (X_test - X_min) / X_range
                X_b_test = np.hstack([X_test_norm, np.ones((X_test_norm.shape[0], 1))])
                pred = X_b_test @ w
                return np.where(np.isfinite(pred), pred, 0.0)
            return predict
        except Exception:
            # Ultimate fallback: mean
            y_mean = np.mean(y)
            def predict(X_test):
                return np.full(X_test.shape[0], y_mean)
            return predict
    
    # Create predict function
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=float)
        X_test_norm = (X_test - X_min) / X_range
        K_test = compute_kernel(X_test_norm, X_norm, freqs)
        pred = K_test @ alpha
        pred = np.where(np.isfinite(pred), pred, 0.0)
        return pred
    
    return predict
# EVOLVE-BLOCK-END
