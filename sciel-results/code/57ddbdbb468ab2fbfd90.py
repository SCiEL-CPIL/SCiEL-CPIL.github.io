import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 320.0

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Standardize X to make kernel parameters more interpretable and stable
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std < 1e-8] = 1.0  # Avoid division by zero
    X_norm = (X - X_mean) / X_std

    def time_left():
        return DEADLINE - time.time()

    # Grid search for hyperparameters
    # Length scales (l) in standardized space
    # Since X is standardized, typical distances are O(1).
    # Small l -> local fit, large l -> global fit.
    l_grid = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    alpha_grid = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0]

    best_loocv_mse = np.inf
    best_l = 1.0
    best_alpha = 1e-4

    # Precompute pairwise distances squared for efficiency?
    # Or just compute kernel for each l.
    
    for l in l_grid:
        if time.time() > DEADLINE - 5:
            break
        # Compute RBF Kernel K
        # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i . x_j
        X_sq = np.sum(X_norm ** 2, axis=1)
        dist_sq = X_sq[:, np.newaxis] + X_sq[np.newaxis, :] - 2 * (X_norm @ X_norm.T)
        # Ensure non-negative due to numerical errors
        np.maximum(dist_sq, 0, out=dist_sq)
        
        K = np.exp(-dist_sq / (2 * l * l))
        
        # Add small diagonal for numerical stability
        K += 1e-10 * np.eye(n)

        for alpha in alpha_grid:
            if time.time() > DEADLINE - 5:
                break
            try:
                # Solve (K + alpha I) w = y
                # Use Cholesky for stability
                A = K + alpha * np.eye(n)
                L = np.linalg.cholesky(A)
                
                # w = L^{-T} L^{-1} y
                # Solve L z = y
                z = np.linalg.solve(L, y)
                # Solve L^T w = z
                w = np.linalg.solve(L.T, z)
                
                # Predictions
                y_pred = K @ w
                
                # LOOCV MSE
                # e_i = (y_i - y_pred_i) / (1 - K_ii_alpha)
                # where K_ii_alpha is the i-th diagonal element of (K + alpha I)^{-1}
                # (K + alpha I)^{-1} = L^{-T} L^{-1}
                # The diagonal elements are sum of squares of rows of L^{-1}
                # Let's compute L_inv = np.linalg.inv(L)
                # This is O(N^3) but N=200, so it's fast.
                L_inv = np.linalg.inv(L)
                diag_inv = np.sum(L_inv ** 2, axis=1)
                
                # Avoid division by zero
                denom = 1 - diag_inv
                denom[np.abs(denom) < 1e-8] = 1e-8
                
                loocv_errors = (y - y_pred) / denom
                loocv_mse = np.mean(loocv_errors ** 2)
                
                if loocv_mse < best_loocv_mse:
                    best_loocv_mse = loocv_mse
                    best_l = l
                    best_alpha = alpha
                    # Store best w? No, we need to refit with the chosen params to be sure, 
                    # or just use this w if we track it.
                    best_w = w
            except:
                continue

    # If no hyperparameter worked, fallback to simple linear
    if not hasattr(fit, '_best_w') and 'best_w' not in locals():
        # Fallback: Ridge Regression on linear features
        X_lin = np.column_stack([np.ones(n), X_norm])
        alpha_fallback = 1.0
        A = X_lin.T @ X_lin + alpha_fallback * np.eye(d+1)
        b = X_lin.T @ y
        best_w = np.linalg.solve(A, b)
        # Need to adapt predict function for this case
        use_linear_fallback = True
    else:
        use_linear_fallback = False

    # Final fit with best hyperparameters to ensure w is optimal for full dataset
    if not use_linear_fallback:
        X_sq = np.sum(X_norm ** 2, axis=1)
        dist_sq = X_sq[:, np.newaxis] + X_sq[np.newaxis, :] - 2 * (X_norm @ X_norm.T)
        np.maximum(dist_sq, 0, out=dist_sq)
        K = np.exp(-dist_sq / (2 * best_l * best_l))
        K += 1e-10 * np.eye(n)
        A = K + best_alpha * np.eye(n)
        L = np.linalg.cholesky(A)
        z = np.linalg.solve(L, y)
        best_w = np.linalg.solve(L.T, z)

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        
        X_test_norm = (X_test - X_mean) / X_std
        
        if use_linear_fallback:
            X_test_lin = np.column_stack([np.ones(len(X_test)), X_test_norm])
            return X_test_lin @ best_w
        else:
            # Compute kernel between test points and training points
            X_train_norm = X_norm
            # dist_sq_test_train[i, j] = ||x_test_i - x_train_j||^2
            X_test_sq = np.sum(X_test_norm ** 2, axis=1)
            X_train_sq = np.sum(X_train_norm ** 2, axis=1)
            dist_sq = X_test_sq[:, np.newaxis] + X_train_sq[np.newaxis, :] - 2 * (X_test_norm @ X_train_norm.T)
            np.maximum(dist_sq, 0, out=dist_sq)
            
            K_test = np.exp(-dist_sq / (2 * best_l * best_l))
            return K_test @ best_w

    return predict
