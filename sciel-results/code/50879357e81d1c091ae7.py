"""Symbolic regression with feature expansion + sparse selection ensemble."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

# Global time tracker
_total_start = None
_time_limit = 310.0  # Leave 20s safety margin before 330s kill

def _elapsed():
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    return time.time() - _total_start

def _remaining():
    return _time_limit - _elapsed()

def _build_feature_matrix(X, max_poly_degree=4, include_trig=True, include_interactions=True, max_features=500):
    """Build a rich feature matrix from input X.
    
    Args:
        X: (n, d) input array
        max_poly_degree: maximum polynomial degree
        include_trig: whether to include sin/cos features
        include_interactions: whether to include pairwise interactions
        max_features: maximum number of features to keep
    
    Returns:
        (n, p) feature matrix
    """
    n, d = X.shape
    features = [np.ones(n)]  # bias
    
    # Polynomial terms for each dimension
    for deg in range(1, max_poly_degree + 1):
        for j in range(d):
            features.append(X[:, j] ** deg)
    
    # Pairwise interactions (products of different dims)
    if include_interactions:
        for i in range(d):
            for j in range(i + 1, d):
                features.append(X[:, i] * X[:, j])
                # Also include squared interactions
                if max_poly_degree >= 3:
                    features.append((X[:, i] * X[:, j]) ** 2)
    
    # Trig features
    if include_trig:
        for j in range(d):
            features.append(np.sin(X[:, j]))
            features.append(np.cos(X[:, j]))
            if max_poly_degree >= 3:
                features.append(np.sin(2 * X[:, j]))
                features.append(np.cos(2 * X[:, j]))
    
    # Exponential features (clipped to avoid overflow)
    for j in range(d):
        features.append(np.exp(np.clip(X[:, j], -10, 10)))
    
    # Log features
    for j in range(d):
        features.append(np.log1p(np.abs(X[:, j])))
    
    # Square root features
    for j in range(d):
        features.append(np.sqrt(np.abs(X[:, j])))
    
    # Reciprocal features
    for j in range(d):
        safe = np.where(np.abs(X[:, j]) > 1e-6, X[:, j], 1e6)
        features.append(1.0 / safe)
    
    # Build matrix
    F = np.column_stack(features)
    
    # Handle any NaN/inf
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    
    # If too many features, select top variance features
    if F.shape[1] > max_features:
        variances = F.var(axis=0)
        top_idx = np.argsort(variances)[::-1][:max_features]
        F = F[:, top_idx]
    
    # Standardize features for better conditioning
    F_mean = F.mean(axis=0)
    F_std = F.std(axis=0)
    F_std[F_std < 1e-10] = 1.0
    F_norm = (F - F_mean) / F_std
    
    return F_norm, F_mean, F_std

def _ridge_solve(A, b, lam):
    """Solve ridge regression (A + lam*I) w = b."""
    n = A.shape[0]
    try:
        w = np.linalg.solve(A + lam * np.eye(n), b)
        return w
    except np.linalg.LinAlgError:
        try:
            w, *_ = np.linalg.lstsq(A + lam * np.eye(n), b, rcond=None)
            return w
        except Exception:
            return np.zeros(n)

def _sparse_select(F_train, y_train, F_val, y_val, max_terms=15, ridge_lambdas=None):
    """Forward selection with ridge regression on selected features.
    
    Args:
        F_train: (n_train, p) training features
        y_train: (n_train,) training targets
        F_val: (n_val, p) validation features
        y_val: (n_val,) validation targets
        max_terms: maximum number of features to select
        ridge_lambdas: list of ridge parameters to try
    
    Returns:
        selected_indices: list of selected feature indices
        best_w: weights for selected features
        best_lam: best ridge lambda
        best_val_mse: best validation MSE
    """
    if ridge_lambdas is None:
        ridge_lambdas = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
    
    n_train, p = F_train.shape
    selected = [0]  # Always include bias
    best_val_mse = np.inf
    best_w = None
    best_lam = None
    
    # Pre-compute Gram matrix for speed
    G = F_train.T @ F_train  # (p, p)
    
    for step in range(max_terms):
        if _remaining() < 2.0:
            break
        
        best_step_mse = np.inf
        best_new_idx = None
        best_step_w = None
        best_step_lam = None
        
        # Try adding each unselected feature
        for j in range(1, p):
            if j in selected:
                continue
            
            new_selected = selected + [j]
            F_sub = F_train[:, new_selected]
            F_val_sub = F_val[:, new_selected]
            
            # Try different ridge lambdas
            for lam in ridge_lambdas:
                w = _ridge_solve(F_sub.T @ F_sub, F_sub.T @ y_train, lam)
                pred_val = F_val_sub @ w
                mse = np.mean((pred_val - y_val) ** 2)
                
                if mse < best_step_mse:
                    best_step_mse = mse
                    best_new_idx = j
                    best_step_w = w
                    best_step_lam = lam
        
        if best_new_idx is None or best_step_mse >= best_val_mse:
            break
        
        selected.append(best_new_idx)
        best_val_mse = best_step_mse
        best_w = best_step_w
        best_lam = best_step_lam
    
    return selected, best_w, best_lam, best_val_mse

def fit(X_train, y_train):
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    
    n, d = X_train.shape
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    
    # Normalize inputs
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X - X_mean) / X_std
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Split for validation
    np.random.seed(42)
    idx = np.arange(n)
    np.random.shuffle(idx)
    val_size = max(20, int(0.15 * n))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    
    # Try multiple feature expansion configurations
    configs = [
        {'max_poly_degree': 3, 'include_trig': True, 'include_interactions': True, 'max_features': 300},
        {'max_poly_degree': 4, 'include_trig': True, 'include_interactions': True, 'max_features': 400},
        {'max_poly_degree': 2, 'include_trig': True, 'include_interactions': True, 'max_features': 250},
        {'max_poly_degree': 3, 'include_trig': False, 'include_interactions': True, 'max_features': 300},
        {'max_poly_degree': 4, 'include_trig': True, 'include_interactions': False, 'max_features': 300},
    ]
    
    best_overall_mse = np.inf
    best_overall_config = None
    best_overall_selected = None
    best_overall_w = None
    best_overall_lam = None
    best_overall_F_norm = None
    best_overall_F_mean = None
    best_overall_F_std = None
    
    for cfg in configs:
        if _remaining() < 10.0:
            break
        
        try:
            F_norm, F_mean, F_std = _build_feature_matrix(X_norm, **cfg)
            F_tr = F_norm[train_idx]
            F_val = F_norm[val_idx]
            y_tr = y_norm[train_idx]
            y_val = y_norm[val_idx]
            
            selected, w, lam, val_mse = _sparse_select(
                F_tr, y_tr, F_val, y_val,
                max_terms=12,
                ridge_lambdas=[1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1]
            )
            
            if val_mse < best_overall_mse:
                best_overall_mse = val_mse
                best_overall_config = cfg
                best_overall_selected = selected
                best_overall_w = w
                best_overall_lam = lam
                best_overall_F_norm = F_norm
                best_overall_F_mean = F_mean
                best_overall_F_std = F_std
        except Exception:
            continue
    
    # If sparse selection didn't work well, try pure ridge on all features
    if best_overall_mse > 0.5 or best_overall_selected is None:
        try:
            F_norm, F_mean, F_std = _build_feature_matrix(X_norm, max_poly_degree=3, include_trig=True, include_interactions=True, max_features=300)
            F_tr = F_norm[train_idx]
            F_val = F_norm[val_idx]
            y_tr = y_norm[train_idx]
            y_val = y_norm[val_idx]
            
            best_ridge_mse = np.inf
            best_ridge_w = None
            best_ridge_lam = None
            
            for lam in [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0]:
                w = _ridge_solve(F_tr.T @ F_tr, F_tr.T @ y_tr, lam)
                pred_val = F_val @ w
                mse = np.mean((pred_val - y_val) ** 2)
                if mse < best_ridge_mse:
                    best_ridge_mse = mse
                    best_ridge_w = w
                    best_ridge_lam = lam
            
            if best_ridge_mse < best_overall_mse:
                best_overall_mse = best_ridge_mse
                best_overall_selected = list(range(F_norm.shape[1]))
                best_overall_w = best_ridge_w
                best_overall_lam = best_ridge_lam
                best_overall_F_norm = F_norm
                best_overall_F_mean = F_mean
                best_overall_F_std = F_std
        except Exception:
            pass
    
    # Final fit on ALL data
    if best_overall_selected is not None and best_overall_F_norm is not None:
        F_all = best_overall_F_norm
        F_sub = F_all[:, best_overall_selected]
        
        if _remaining() > 2.0:
            try:
                final_w = _ridge_solve(F_sub.T @ F_sub, F_sub.T @ y_norm, best_overall_lam)
            except Exception:
                final_w = best_overall_w
        else:
            final_w = best_overall_w
    else:
        # Fallback: simple linear regression
        try:
            A = np.column_stack([np.ones(n), X_norm])
            final_w, *_ = np.linalg.lstsq(A, y_norm, rcond=None)
            # Reconstruct as feature-based
            F_all = np.column_stack([np.ones(n), X_norm])
            best_overall_selected = list(range(F_all.shape[1]))
            best_overall_F_norm = F_all
            best_overall_F_mean = np.zeros(F_all.shape[1])
            best_overall_F_std = np.ones(F_all.shape[1])
        except Exception:
            final_w = np.zeros(d + 1)
            F_all = np.column_stack([np.ones(n), X_norm])
            best_overall_selected = list(range(F_all.shape[1]))
            best_overall_F_norm = F_all
            best_overall_F_mean = np.zeros(F_all.shape[1])
            best_overall_F_std = np.ones(F_all.shape[1])
    
    # Capture for closure
    F_norm_all = best_overall_F_norm
    F_mean = best_overall_F_mean
    F_std = best_overall_F_std
    selected = best_overall_selected
    w_final = final_w if 'final_w' in dir() else best_overall_w
    
    def predict(X_test):
        X_t = X_test.astype(np.float64)
        X_t_norm = (X_t - X_mean) / X_std
        
        # Rebuild feature matrix for test
        F_test, _, _ = _build_feature_matrix(X_t_norm, **best_overall_config)
        F_test_sub = F_test[:, selected]
        
        y_pred_norm = F_test_sub @ w_final
        return y_pred_norm * y_std + y_mean
    
    return predict
# EVOLVE-BLOCK-END
