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
    
    def ridge_solve(F, y, alpha):
        """Solve ridge regression: (F^T F + alpha*I) w = F^T y"""
        n_feat = F.shape[1]
        try:
            A = F.T @ F + alpha * np.eye(n_feat)
            b = F.T @ y
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(F + alpha * np.eye(F.shape[0])) @ y

    def loo_cv_score(F, y, alpha):
        """Leave-one-out CV score using the hat matrix trick for ridge regression."""
        if time.time() > deadline:
            return np.inf
        n = F.shape[0]
        w = ridge_solve(F, y, alpha)
        Fw = F @ w
        # Hat matrix H = F(F'F + alpha I)^{-1}F'
        # LOO prediction: y_i - (H_ii / (1 - H_ii)) * (y_i - Fw_i)
        try:
            A = F.T @ F + alpha * np.eye(F.shape[1])
            A_inv = np.linalg.inv(A)
            H_diag = np.sum((F * (F @ A_inv)), axis=1)
            loo_preds = Fw - H_diag / (1 - H_diag) * (y - Fw)
        except np.linalg.LinAlgError:
            return np.inf
        return np.mean((y - loo_preds)**2)

    # Fast feature screening: use LASSO-like screening with a single pass
    # First, do a quick ridge fit with all features to get initial weights
    alpha_screen = 1e-2
    w_screen = ridge_solve(F_train, y_norm, alpha_screen)
    
    # Score each feature by |correlation| with residual
    residual = y_norm - F_train @ w_screen
    n_feat = F_train.shape[1]
    feature_scores = np.zeros(n_feat)
    for j in range(n_feat):
        denom = np.std(F_train[:, j])
        if denom > 1e-10:
            feature_scores[j] = np.abs(np.corrcoef(F_train[:, j], residual)[0, 1])
    
    # Select top features based on score, then do forward selection on a smaller set
    top_k = min(80, n_feat)
    top_indices = np.argsort(feature_scores)[::-1][:top_k]
    top_indices = np.sort(top_indices)
    
    # Ensure intercept is included
    if 0 not in top_indices:
        top_indices = np.insert(top_indices, 0, 0)
    
    F_screen = F_train[:, top_indices]
    remaining_screen = [j for j in range(len(top_indices)) if j != 0]
    selected_screen = [0]
    
    # Forward selection with LOO CV (much faster than k-fold)
    max_fwd = min(35, len(top_indices))
    best_score = loo_cv_score(F_screen[:, selected_screen], y_norm, 1e-3)
    
    for _ in range(max_fwd - 1):
        if time.time() > deadline - 40:
            break
        
        best_new_score = best_score
        best_new_idx = None
        
        for idx in remaining_screen:
            if time.time() > deadline - 30:
                break
            trial = selected_screen + [idx]
            score = loo_cv_score(F_screen[:, trial], y_norm, 1e-3)
            if score < best_new_score:
                best_new_score = score
                best_new_idx = idx
        
        if best_new_idx is None:
            break
        
        selected_screen.append(best_new_idx)
        remaining_screen.remove(best_new_idx)
        best_score = best_new_score
    
    # Map back to original feature indices
    selected_indices = [top_indices[i] for i in selected_screen]
    
    # Final ridge fit with selected features
    F_selected = F_train[:, selected_indices]
    alpha_final = 1e-3
    w_final = ridge_solve(F_selected, y_norm, alpha_final)
    
    # Also fit a simple RBF KRR as an ensemble member
    def rbf_kernel(X1, X2, lengthscale):
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * np.dot(X1, X2.T)
        np.maximum(sq_dists, 0, out=sq_dists)
        return np.exp(-sq_dists / (2 * lengthscale**2))
    
    # Try a few RBF lengthscales and pick best via LOO
    best_rbf_ls = None
    best_rbf_score = np.inf
    for ls in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        if time.time() > deadline - 20:
            break
        K = rbf_kernel(X_norm, X_norm, ls)
        alpha_krr = 1e-2
        try:
            K_reg = K + alpha_krr * np.eye(n_samples)
            w_krr = np.linalg.solve(K_reg, y_norm)
            # LOO via hat matrix
            K_inv = np.linalg.inv(K_reg)
            H_diag = np.sum(K * K_inv, axis=1)
            K_w = K @ w_krr
            loo_preds = K_w - H_diag / (1 - H_diag) * (y_norm - K_w)
            score = np.mean((y_norm - loo_preds)**2)
            if score < best_rbf_score:
                best_rbf_score = score
                best_rbf_ls = ls
        except np.linalg.LinAlgError:
            continue
    
    use_rbf = best_rbf_ls is not None and best_rbf_score < best_score * 1.5
    
    if use_rbf:
        K_train = rbf_kernel(X_norm, X_norm, best_rbf_ls)
        K_reg = K_train + 1e-2 * np.eye(n_samples)
        w_krr = np.linalg.solve(K_reg, y_norm)
    else:
        w_krr = None
    
    # Capture for prediction
    f_X_mean = X_mean
    f_X_std = X_std
    f_selected_indices = selected_indices
    f_w = w_final
    f_y_mean = y_mean
    f_y_std = y_std
    f_n_features = n_features
    f_use_rbf = use_rbf
    f_best_rbf_ls = best_rbf_ls
    f_w_krr = w_krr
    f_X_norm = X_norm
    f_y_norm = y_norm
    f_n_samples = n_samples
    f_best_rbf_score = best_rbf_score
    f_best_score = best_score
    
    def predict(X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X_norm_test = (X - f_X_mean) / f_X_std
        F_test_full = build_feature_library(X_norm_test)
        col_var_test_full = np.var(F_test_full, axis=0) if F_test_full.shape[0] > 1 else np.ones(F_test_full.shape[1])
        valid_cols_full = col_var_test_full > 1e-10
        F_test_filtered = F_test_full[:, valid_cols_full]
        
        # Extract selected features
        F_test_selected = F_test_filtered[:, f_selected_indices]
        pred_feat_norm = F_test_selected @ f_w
        
        if f_use_rbf:
            K_test = rbf_kernel(X_norm_test, f_X_norm, f_best_rbf_ls)
            pred_krr_norm = K_test @ f_w_krr
            # Weighted ensemble based on LOO scores
            w_feat = 1.0 / (f_best_score + 1e-10)
            w_krr = 1.0 / (f_best_rbf_score + 1e-10)
            w_total = w_feat + w_krr
            pred_norm = (w_feat * pred_feat_norm + w_krr * pred_krr_norm) / w_total
        else:
            pred_norm = pred_feat_norm
        
        pred_orig = pred_norm * f_y_std + f_y_mean
        
        return pred_orig

    return predict
