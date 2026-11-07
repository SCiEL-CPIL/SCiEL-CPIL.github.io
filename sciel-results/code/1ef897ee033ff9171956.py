import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin from 330s limit

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

    def rbf_kernel(X1, X2, lengthscale):
        """RBF kernel with given lengthscale (already normalized inputs)."""
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * np.dot(X1, X2.T)
        np.maximum(sq_dists, 0, out=sq_dists)
        return np.exp(-sq_dists / (2 * lengthscale**2))

    def poly_kernel(X1, X2, degree, gamma=1.0):
        """Polynomial kernel."""
        return (gamma * np.dot(X1, X2.T) + 1) ** degree

    def linear_kernel(X1, X2):
        return np.dot(X1, X2.T)

    # Try multiple kernel configurations and pick the best via cross-validation
    def cv_score(kernel_func, alpha_reg):
        """5-fold cross-validation score (lower is better)."""
        if time.time() > deadline:
            return np.inf
        
        n = n_samples
        fold_size = n // 5
        residuals = np.zeros(n)
        
        for fold in range(5):
            if time.time() > deadline:
                return np.inf
            val_idx = np.arange(fold * fold_size, (fold + 1) * fold_size)
            trn_idx = np.setdiff1d(np.arange(n), val_idx)
            
            X_trn = X_norm[trn_idx]
            y_trn = y_norm[trn_idx]
            X_val = X_norm[val_idx]
            
            # Build kernel matrix for training
            K = kernel_func(X_trn, X_trn)
            K[np.diag_indices_from(K)] += alpha_reg
            
            try:
                # Solve (K + alpha*I) w = y
                w = np.linalg.solve(K, y_trn)
                # Predict on validation
                K_val = kernel_func(X_val, X_trn)
                pred_val = K_val @ w
                residuals[val_idx] = y_norm[val_idx] - pred_val
            except np.linalg.LinAlgError:
                return np.inf
        
        return np.mean(residuals**2)

    # Candidate kernel configurations
    candidates = []
    
    # RBF with various lengthscales
    for ls in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]:
        candidates.append(('rbf', ls, lambda X1, X2, ls=ls: rbf_kernel(X1, X2, ls)))
    
    # Polynomial kernels
    for deg in [2, 3, 4, 5]:
        for gamma in [0.5, 1.0, 2.0]:
            candidates.append((f'poly{deg}_{gamma}', 0, lambda X1, X2, d=deg, g=gamma: poly_kernel(X1, X2, d, g)))
    
    # Linear
    candidates.append(('linear', 0, lambda X1, X2: linear_kernel(X1, X2)))
    
    # Combined: RBF + polynomial
    for ls in [0.3, 0.5, 1.0]:
        for deg in [2, 3]:
            candidates.append((f'combo_rbf{ls}_poly{deg}', 0, 
                             lambda X1, X2, ls=ls, d=deg: rbf_kernel(X1, X2, ls) + poly_kernel(X1, X2, d, 1.0)))
    
    # Regularization strengths
    alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0]
    
    best_score = np.inf
    best_kernel_func = None
    best_alpha = None
    best_kernel_name = None
    
    # Evaluate candidates
    for name, _, kfunc in candidates:
        if time.time() > deadline - 30:
            break
        for alpha in alphas:
            if time.time() > deadline - 15:
                break
            score = cv_score(kfunc, alpha)
            if score < best_score:
                best_score = score
                best_kernel_func = kfunc
                best_alpha = alpha
                best_kernel_name = name
    
    # If no kernel worked well, fall back to linear regression
    if best_kernel_func is None:
        X_design = np.hstack([X_train, np.ones((n_samples, 1))])
        try:
            w, _, _, _ = np.linalg.lstsq(X_design, y_train, rcond=None)
            def predict_lin(X):
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                X_pred = np.hstack([X, np.ones((X.shape[0], 1))])
                return X_pred @ w
            return predict_lin
        except Exception:
            def predict_mean(X):
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                return np.full(X.shape[0], y_mean)
            return predict_mean

    # Final fit with best kernel and regularization
    K = best_kernel_func(X_norm, X_norm)
    K[np.diag_indices_from(K)] += best_alpha
    
    try:
        w = np.linalg.solve(K, y_norm)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        w = np.linalg.pinv(K) @ y_norm

    # Capture for prediction
    f_X_mean = X_mean
    f_X_std = X_std
    f_kernel_func = best_kernel_func
    f_w = w
    f_y_mean = y_mean
    f_y_std = y_std
    f_X_norm = X_norm

    def predict(X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X_norm_test = (X - f_X_mean) / f_X_std
        K_test = f_kernel_func(X_norm_test, f_X_norm)
        mean_norm = K_test @ f_w
        mean_orig = mean_norm * f_y_std + f_y_mean
        
        return mean_orig

    return predict
