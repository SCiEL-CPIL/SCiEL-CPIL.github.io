"""Symbolic regression with feature expansion + ensemble of linear models."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

# Global time tracker
_total_start = None
_time_limit = 320.0  # Leave 10s safety margin before 330s kill

def _elapsed():
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    return time.time() - _total_start

def _remaining():
    return _time_limit - _elapsed()

def _build_feature_matrix(X, max_poly_degree=4, include_trig=True, include_interactions=True):
    """
    Build an expanded feature matrix from input X.
    Includes:
    - Constant term
    - Original features
    - Polynomial terms up to max_poly_degree
    - Interaction terms (products of distinct features) if include_interactions
    - Trigonometric terms (sin, cos) if include_trig
    """
    n, d = X.shape
    features = [np.ones(n)]
    
    # Original features
    for j in range(d):
        features.append(X[:, j])
    
    # Polynomial terms
    for deg in range(2, max_poly_degree + 1):
        for j in range(d):
            features.append(X[:, j] ** deg)
            
    # Interaction terms (pairwise for now to keep dimensionality manageable)
    if include_interactions:
        for i in range(d):
            for j in range(i + 1, d):
                features.append(X[:, i] * X[:, j])
                # Maybe higher order interactions if d is small?
                if d <= 3:
                    # Add triple products if d is small
                    for k in range(j + 1, d):
                        features.append(X[:, i] * X[:, j] * X[:, k])

    # Trigonometric terms
    if include_trig:
        for j in range(d):
            features.append(np.sin(X[:, j]))
            features.append(np.cos(X[:, j]))
            # Maybe sin/cos of interactions?
            # features.append(np.sin(X[:, i] * X[:, j])) - might be too many

    return np.column_stack(features)

def _ridge_solve(X, y, lam):
    """Solve Ridge regression problem."""
    n, p = X.shape
    # X^T X + lam I
    XtX = X.T @ X
    Xty = X.T @ y
    A = XtX + lam * np.eye(p)
    # Add small jitter for stability
    A += 1e-10 * np.eye(p)
    w = np.linalg.solve(A, Xty)
    return w

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
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    val_size = max(20, int(0.1 * n))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    
    X_tr = X_norm[train_idx]
    y_tr = y_norm[train_idx]
    X_val = X_norm[val_idx]
    y_val = y_norm[val_idx]
    
    # Define feature configurations to try
    # (max_poly_degree, include_trig, include_interactions)
    feature_configs = [
        (2, False, True),
        (3, False, True),
        (2, True, True),
        (3, True, True),
        (4, False, False), # Pure polynomials
        (2, True, False),  # Poly + Trig
    ]
    
    # Ridge regularization strengths
    ridge_lambdas = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    
    best_val_mse = np.inf
    best_config = None # (feature_config, lam)
    best_weights = None
    best_feature_builder_args = None
    
    # Pre-compute feature matrices for training and validation for each config?
    # It might be faster to compute on the fly if number of configs is small.
    
    for max_deg, use_trig, use_inter in feature_configs:
        if _remaining() < 5.0:
            break
        try:
            # Build features for train and val
            F_tr = _build_feature_matrix(X_tr, max_deg, use_trig, use_inter)
            F_val = _build_feature_matrix(X_val, max_deg, use_trig, use_inter)
            
            # Check for NaNs or Infs
            if not (np.all(np.isfinite(F_tr)) and np.all(np.isfinite(F_val))):
                continue
                
            for lam in ridge_lambdas:
                try:
                    w = _ridge_solve(F_tr, y_tr, lam)
                    pred_val = F_val @ w
                    mse = np.mean((pred_val - y_val) ** 2)
                    if mse < best_val_mse:
                        best_val_mse = mse
                        best_config = (max_deg, use_trig, use_inter, lam)
                        best_weights = w
                        best_feature_builder_args = (max_deg, use_trig, use_inter)
                except np.linalg.LinAlgError:
                    continue
                except Exception:
                    continue
        except Exception:
            continue
            
    # If no config worked, fall back to simple linear
    if best_config is None:
        # Simple linear regression on raw features
        X_tr_aug = np.column_stack([np.ones(len(X_tr)), X_tr])
        X_val_aug = np.column_stack([np.ones(len(X_val)), X_val])
        try:
            w, *_ = np.linalg.lstsq(X_tr_aug, y_tr, rcond=None)
            pred_val = X_val_aug @ w
            mse = np.mean((pred_val - y_val) ** 2)
            if mse < best_val_mse:
                best_val_mse = mse
                best_config = (1, False, False, 0.0) # Marker for linear
                best_weights = w
                best_feature_builder_args = (1, False, False)
        except Exception:
            # Absolute fallback: mean prediction
            best_config = (0, False, False, 0.0)
            best_weights = np.array([1.0]) # Constant term only
            best_feature_builder_args = (0, False, False)
            best_val_mse = np.mean((y_val - y_tr.mean())**2)

    # Final fit on ALL data with the best configuration
    if best_config is not None:
        max_deg, use_trig, use_inter, lam = best_config
        
        if max_deg == 0:
            # Constant model
            final_weights = np.array([np.mean(y_norm)])
        else:
            F_full = _build_feature_matrix(X_norm, max_deg, use_trig, use_inter)
            try:
                final_weights = _ridge_solve(F_full, y_norm, lam)
            except Exception:
                # Fallback to training weights if full fit fails
                final_weights = best_weights
    
    # Determine prediction method
    def predict(X_test):
        X_t = X_test.astype(np.float64)
        X_t_norm = (X_t - X_mean) / X_std
        
        max_deg, use_trig, use_inter, lam = best_config
        
        if max_deg == 0:
            # Constant model
            y_pred_norm = np.full(len(X_t), final_weights[0])
        else:
            F_test = _build_feature_matrix(X_t_norm, max_deg, use_trig, use_inter)
            y_pred_norm = F_test @ final_weights
            
        return y_pred_norm * y_std + y_mean
    
    return predict
# EVOLVE-BLOCK-END
