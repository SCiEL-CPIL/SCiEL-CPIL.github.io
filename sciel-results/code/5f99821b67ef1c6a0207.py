import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 295.0  # Safety margin from 330s limit

    n_samples, n_features = X_train.shape
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)

    if y_std < 1e-10:
        def predict_const(X):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return np.full(X.shape[0], y_mean)
        return predict_const

    # Normalize X for better conditioning
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X_train - X_mean) / X_std

    # Normalize y
    y_norm = (y_train - y_mean) / y_std

    def build_feature_library(X):
        """Build a rich library of basis functions from input X (already normalized)."""
        features = [np.ones((X.shape[0], 1))]
        
        # Linear terms
        for j in range(X.shape[1]):
            features.append(X[:, j:j+1])
        
        # Quadratic terms
        for j in range(X.shape[1]):
            features.append(X[:, j:j+1] ** 2)
        
        # Cross terms (quadratic)
        for i in range(X.shape[1]):
            for j in range(i+1, X.shape[1]):
                features.append(X[:, i:i+1] * X[:, j:j+1])
        
        # Cubic terms
        for j in range(X.shape[1]):
            features.append(X[:, j:j+1] ** 3)
        
        # Cross cubic terms
        for i in range(X.shape[1]):
            for j in range(i+1, X.shape[1]):
                features.append(X[:, i:i+1] ** 2 * X[:, j:j+1])
                features.append(X[:, i:i+1] * X[:, j:j+1] ** 2)
        
        # Sine/cosine terms (periodic functions)
        for j in range(X.shape[1]):
            for freq in [0.5, 1.0, 2.0, 3.0]:
                features.append(np.sin(freq * X[:, j:j+1]))
                features.append(np.cos(freq * X[:, j:j+1]))
        
        # Exponential terms
        for j in range(X.shape[1]):
            for scale in [0.5, 1.0, 2.0]:
                features.append(np.exp(-scale * X[:, j:j+1] ** 2))
        
        # RBF-like terms with different centers
        n_rbf_centers = min(10, X.shape[0])
        if n_rbf_centers >= 3:
            # Use evenly spaced points as RBF centers
            if X.shape[1] == 1:
                centers = np.linspace(-1.5, 1.5, n_rbf_centers)
            else:
                # For multi-dim, use random-ish centers from data
                indices = np.random.choice(X.shape[0], n_rbf_centers, replace=False)
                centers = X[indices]
            
            for ls in [0.3, 0.7, 1.5]:
                for c in centers:
                    if X.shape[1] == 1:
                        dist_sq = (X[:, 0] - c) ** 2
                    else:
                        dist_sq = np.sum((X - c) ** 2, axis=1)
                    features.append(np.exp(-dist_sq / (2 * ls ** 2)).reshape(-1, 1))
        
        return np.hstack(features)

    # Build feature library
    F_train = build_feature_library(X_norm)
    
    # Remove columns with very small variance
    col_var = np.var(F_train, axis=0)
    valid_cols = col_var > 1e-10
    F_train = F_train[:, valid_cols]
    
    # Iterative feature selection using LASSO-like approach with ridge regularization
    # Use forward selection with cross-validation
    
    def ridge_solve(F, y, alpha):
        """Solve ridge regression: (F^T F + alpha*I) w = F^T y"""
        n_feat = F.shape[1]
        try:
            A = F.T @ F + alpha * np.eye(n_feat)
            b = F.T @ y
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(F + alpha * np.eye(F.shape[0])) @ y

    def cv_ridge_score(F, y, alpha, n_folds=5):
        """Cross-validation score for ridge regression."""
        if time.time() > deadline:
            return np.inf
        
        n = F.shape[0]
        fold_size = n // n_folds
        residuals = np.zeros(n)
        
        for fold in range(n_folds):
            if time.time() > deadline:
                return np.inf
            val_idx = np.arange(fold * fold_size, (fold + 1) * fold_size)
            trn_idx = np.setdiff1d(np.arange(n), val_idx)
            
            F_trn = F[trn_idx]
            y_trn = y[trn_idx]
            F_val = F[val_idx]
            
            w = ridge_solve(F_trn, y_trn, alpha)
            pred_val = F_val @ w
            residuals[val_idx] = y[val_idx] - pred_val
        
        return np.mean(residuals**2)

    # Forward feature selection
    max_features = min(50, F_train.shape[1])
    selected_indices = [0]  # Always include intercept
    remaining_indices = list(range(1, F_train.shape[1]))
    
    best_cv_score = cv_ridge_score(F_train[:, selected_indices], y_norm, 1e-3)
    
    for _ in range(max_features - 1):
        if time.time() > deadline - 30:
            break
        
        best_new_score = best_cv_score
        best_new_idx = None
        
        for idx in remaining_indices:
            if time.time() > deadline - 20:
                break
            
            trial_indices = selected_indices + [idx]
            score = cv_ridge_score(F_train[:, trial_indices], y_norm, 1e-3)
            
            if score < best_new_score:
                best_new_score = score
                best_new_idx = idx
        
        if best_new_idx is None:
            break
        
        selected_indices.append(best_new_idx)
        remaining_indices.remove(best_new_idx)
        best_cv_score = best_new_score
    
    # Final ridge fit with selected features
    F_selected = F_train[:, selected_indices]
    alpha_final = 1e-3
    w_final = ridge_solve(F_selected, y_norm, alpha_final)
    
    # Capture for prediction
    f_X_mean = X_mean
    f_X_std = X_std
    f_selected_indices = selected_indices
    f_w = w_final
    f_y_mean = y_mean
    f_y_std = y_std
    f_n_features = n_features
    f_max_features = len(selected_indices)
    
    def predict(X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X_norm_test = (X - f_X_mean) / f_X_std
        F_test = build_feature_library(X_norm_test)
        
        # Apply same column filtering
        col_var_test = np.var(F_test, axis=0) if F_test.shape[0] > 1 else np.ones(F_test.shape[1])
        valid_cols_test = col_var_test > 1e-10
        F_test = F_test[:, valid_cols_test]
        
        # Select the same columns as during training
        # Need to map indices through the valid_cols filter
        # Since we're using the same X_norm computation, the valid cols should be consistent
        # But to be safe, let's just use all features and re-derive
        F_test_full = build_feature_library(X_norm_test)
        col_var_test_full = np.var(F_test_full, axis=0) if F_test_full.shape[0] > 1 else np.ones(F_test_full.shape[1])
        valid_cols_full = col_var_test_full > 1e-10
        F_test_filtered = F_test_full[:, valid_cols_full]
        
        # Extract selected features
        F_test_selected = F_test_filtered[:, f_selected_indices]
        
        pred_norm = F_test_selected @ f_w
        pred_orig = pred_norm * f_y_std + f_y_mean
        
        return pred_orig

    return predict
