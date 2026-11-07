"""Symbolic regression with adaptive feature expansion + sparse linear ensemble."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

# Global time tracker
_total_start = None
_time_limit = 315.0  # Leave 15s safety margin before 330s kill

def _elapsed():
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    return time.time() - _total_start

def _remaining():
    return _time_limit - _elapsed()

def _build_feature_matrix(X, max_degree=5, include_interactions=True):
    """
    Build a rich feature matrix from input X.
    Features include:
    - All monomials up to max_degree (with exponents summing to <= max_degree)
    - sin, cos, tanh, exp(-x^2), log(1+x), 1/(1+x) of each input
    - Interactions between different features (up to degree 2 of transformed features)
    """
    n, d = X.shape
    features = [np.ones((n, 1))]
    
    # Basic features
    base_features = [X[:, i:i+1] for i in range(d)]
    
    # Polynomial monomials up to max_degree
    for deg in range(2, max_degree + 1):
        # Generate all monomials of degree deg
        # For simplicity, we'll generate all combinations of exponents that sum to deg
        # This can be expensive, so we limit it
        if d <= 3:
            # Generate all monomials of total degree up to max_degree
            for exponents in _generate_monomials(d, deg):
                feat = np.ones((n, 1))
                for i, exp in enumerate(exponents):
                    if exp > 0:
                        feat = feat * (X[:, i:i+1] ** exp)
                features.append(feat)
        else:
            # For higher dimensions, just do simple monomials
            for i in range(d):
                for j in range(i+1, d) if deg == 2 else []:
                    features.append(X[:, i:i+1] * X[:, j:j+1])
            if deg == 2:
                for i in range(d):
                    features.append(X[:, i:i+1] ** 2)
    
    # Transformed features
    transforms = [
        lambda x: np.sin(x),
        lambda x: np.cos(x),
        lambda x: np.tanh(x),
        lambda x: np.exp(-x**2 / 2),
        lambda x: np.log1p(np.abs(x)),
        lambda x: 1.0 / (1.0 + np.abs(x)),
        lambda x: x / (1.0 + np.abs(x)),
    ]
    
    for t in transforms:
        for i in range(d):
            try:
                feat = t(X[:, i:i+1])
                # Check for NaN/Inf
                if np.all(np.isfinite(feat)):
                    features.append(feat)
            except Exception:
                pass
    
    # Interactions between base features and transformed features
    if include_interactions and d <= 3:
        base_list = [X[:, i:i+1] for i in range(d)]
        for i in range(d):
            for j in range(i+1, d):
                # Product of two different inputs
                feat = X[:, i:i+1] * X[:, j:j+1]
                features.append(feat)
                # Also with transforms
                for t in transforms[:3]:
                    try:
                        feat2 = t(X[:, i:i+1]) * X[:, j:j+1]
                        if np.all(np.isfinite(feat2)):
                            features.append(feat2)
                        feat3 = X[:, i:i+1] * t(X[:, j:j+1])
                        if np.all(np.isfinite(feat3)):
                            features.append(feat3)
                    except Exception:
                        pass
    
    # Stack all features
    F = np.hstack(features)
    
    # Remove columns that are all zeros or have NaN/Inf
    valid_cols = np.all(np.isfinite(F), axis=0) & (np.abs(F).sum(axis=0) > 1e-10)
    F = F[:, valid_cols]
    
    # Standardize features
    F_mean = F.mean(axis=0)
    F_std = F.std(axis=0)
    F_std[F_std < 1e-10] = 1.0
    F_norm = (F - F_mean) / F_std
    
    return F_norm, F_mean, F_std, len(features)

def _generate_monomials(d, degree):
    """Generate all exponent tuples of length d that sum to exactly degree."""
    if d == 1:
        yield (degree,)
        return
    for i in range(degree + 1):
        for rest in _generate_monomials(d - 1, degree - i):
            yield (i,) + rest

def _ridge_solve(A, b, lam):
    """Solve ridge regression: (A^T A + lam*I) w = A^T b"""
    n, p = A.shape
    if p > n:
        # More features than samples, use SVD
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        # Regularized solution
        s_reg = s / (s**2 + lam)
        w = Vt.T @ (s_reg * (U.T @ b))
        return w
    else:
        ATA = A.T @ A + lam * np.eye(p)
        ATb = A.T @ b
        try:
            w = np.linalg.solve(ATA, ATb)
            return w
        except np.linalg.LinAlgError:
            w, *_ = np.linalg.lstsq(ATA, ATb, rcond=None)
            return w

def fit(X_train, y_train):
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    
    n, d = X_train.shape
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Try multiple feature expansion strategies
    strategies = [
        (3, True),   # max_degree=3, interactions=True
        (4, True),   # max_degree=4, interactions=True
        (5, True),   # max_degree=5, interactions=True
        (3, False),  # max_degree=3, interactions=False
        (2, True),   # max_degree=2, interactions=True
    ]
    
    # Split for validation
    np.random.seed(42)
    idx = np.arange(n)
    np.random.shuffle(idx)
    val_size = max(20, int(0.1 * n))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    
    best_val_mse = np.inf
    best_strategy = None
    best_w = None
    best_F_mean = None
    best_F_std = None
    
    ridge_lambdas = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    
    for max_deg, use_int in strategies:
        if _remaining() < 10.0:
            break
        
        try:
            F_train_full, F_mean, F_std, n_feats = _build_feature_matrix(X, max_deg, use_int)
            F_tr = F_train_full[train_idx]
            F_val = F_train_full[val_idx]
            y_tr = y_norm[train_idx]
            y_val = y_norm[val_idx]
            
            for lam in ridge_lambdas:
                if _remaining() < 5.0:
                    break
                try:
                    w = _ridge_solve(F_tr, y_tr, lam)
                    pred_val = F_val @ w
                    mse = np.mean((pred_val - y_val) ** 2)
                    if mse < best_val_mse:
                        best_val_mse = mse
                        best_strategy = (max_deg, use_int)
                        best_w = w
                        best_F_mean = F_mean
                        best_F_std = F_std
                except Exception:
                    continue
        except Exception:
            continue
    
    # If no good solution found, fall back to simple linear
    if best_w is None:
        try:
            X_aug = np.column_stack([np.ones(n), X])
            X_aug_mean = X_aug.mean(axis=0)
            X_aug_std = X_aug.std(axis=0)
            X_aug_std[X_aug_std < 1e-10] = 1.0
            X_aug_norm = (X_aug - X_aug_mean) / X_aug_std
            
            X_tr_aug = X_aug_norm[train_idx]
            X_val_aug = X_aug_norm[val_idx]
            
            best_w = _ridge_solve(X_tr_aug, y_norm[train_idx], 1e-4)
            best_F_mean = X_aug_mean
            best_F_std = X_aug_std
            best_strategy = ('linear_fallback',)
        except Exception:
            # Ultimate fallback: predict mean
            def predict(X_test):
                return np.full(len(X_test), y_mean)
            return predict
    
    # Final fit on ALL data with the best strategy
    try:
        if best_strategy[0] == 'linear_fallback':
            X_aug = np.column_stack([np.ones(n), X])
            X_aug_norm = (X_aug - best_F_mean) / best_F_std
            final_w = _ridge_solve(X_aug_norm, y_norm, 1e-4)
        else:
            max_deg, use_int = best_strategy
            F_full, _, _, _ = _build_feature_matrix(X, max_deg, use_int)
            # Use the same F_mean and F_std from validation
            F_full_norm = (F_full - best_F_mean) / best_F_std
            # Re-derive F_mean and F_std from full data for consistency
            F_full_mean = F_full.mean(axis=0)
            F_full_std = F_full.std(axis=0)
            F_full_std[F_full_std < 1e-10] = 1.0
            F_full_norm = (F_full - F_full_mean) / F_full_std
            final_w = _ridge_solve(F_full_norm, y_norm, 1e-4)
            best_F_mean = F_full_mean
            best_F_std = F_full_std
    except Exception:
        final_w = best_w
    
    def predict(X_test):
        X_t = X_test.astype(np.float64)
        
        try:
            if best_strategy[0] == 'linear_fallback':
                X_t_aug = np.column_stack([np.ones(len(X_t)), X_t])
                X_t_norm = (X_t_aug - best_F_mean) / best_F_std
            else:
                max_deg, use_int = best_strategy
                F_t, _, _, _ = _build_feature_matrix(X_t, max_deg, use_int)
                F_t_norm = (F_t - best_F_mean) / best_F_std
            
            y_pred_norm = F_t_norm @ final_w
            y_pred = y_pred_norm * y_std + y_mean
            
            # Ensure finite values
            y_pred = np.where(np.isfinite(y_pred), y_pred, y_mean)
            return y_pred
        except Exception:
            return np.full(len(X_test), y_mean)
    
    return predict
# EVOLVE-BLOCK-END
