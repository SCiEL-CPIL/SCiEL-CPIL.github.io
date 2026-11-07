"""Combined approach: rich feature basis with ridge regression and model selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 300  # Safety margin before 330s kill


def _time_left():
    return _DEADLINE - time.time()


def _build_basis(X):
    """Build a rich feature basis: polynomials + non-linear + interactions."""
    d = X.shape[1]
    feats = [np.ones(X.shape[0])]
    
    # Linear terms
    for j in range(d):
        feats.append(X[:, j])
    
    # Quadratic terms (including cross terms)
    for i in range(d):
        for j in range(i, d):
            feats.append(X[:, i] * X[:, j])
    
    # Cubic terms (only for small d to avoid explosion)
    if d <= 3:
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    feats.append(X[:, i] * X[:, j] * X[:, k])
    
    # Non-linear single-variable functions
    for i in range(d):
        x = X[:, i]
        # Trigonometric
        feats.append(np.sin(x))
        feats.append(np.cos(x))
        feats.append(np.sin(2 * x))
        feats.append(np.cos(2 * x))
        # Exponential (scaled to avoid overflow for x in [1,10])
        feats.append(np.exp(-0.1 * x))
        feats.append(np.exp(-0.3 * x))
        feats.append(np.exp(-0.5 * x))
        # Logarithmic
        feats.append(np.log1p(x))
        feats.append(np.log(x + 1e-10))
        # Power functions
        feats.append(np.sqrt(x))
        feats.append(x ** 0.75)
        # Hyperbolic
        feats.append(np.tanh(x))
        feats.append(np.tanh(0.5 * x))
    
    # Pairwise interactions of non-linear with linear
    for i in range(d):
        for j in range(d):
            if i != j:
                feats.append(np.sin(X[:, i]) * X[:, j])
                feats.append(np.cos(X[:, i]) * X[:, j])
                feats.append(np.sqrt(X[:, i]) * X[:, j])
                feats.append(np.log1p(X[:, i]) * X[:, j])
    
    F = np.column_stack(feats)
    # Handle NaN/Inf
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F


def _fit_ridge(F, y, lam):
    """Fit ridge regression: (F^T F + lam*I) w = F^T y."""
    n_features = F.shape[1]
    XTX = F.T @ F
    XTy = F.T @ y
    A = XTX + lam * np.eye(n_features)
    try:
        w = np.linalg.solve(A, XTy)
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(F, y, rcond=None)
    return w


def fit(X_train, y_train):
    start_time = time.time()
    
    F_train = _build_basis(X_train)
    n = len(y_train)
    n_features = F_train.shape[1]
    
    # Standardize features for numerical stability
    feat_mean = F_train.mean(axis=0)
    feat_std = F_train.std(axis=0)
    feat_std[feat_std < 1e-10] = 1.0
    F_std = (F_train - feat_mean) / feat_std
    
    # Cross-validation to select lambda
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    val_size = max(20, n // 5)
    val_idx = indices[:val_size]
    tr_idx = indices[val_size:]
    
    F_tr = F_std[tr_idx]
    F_val = F_std[val_idx]
    y_tr = y_train[tr_idx]
    y_val = y_train[val_idx]
    
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    best_lambda = 1e-4
    best_mse = float('inf')
    
    for lam in lambdas:
        if _time_left() < 10:
            break
        w = _fit_ridge(F_tr, y_tr, lam)
        pred_val = F_val @ w
        mse = np.mean((pred_val - y_val) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_lambda = lam
    
    # Refine on full training set with best lambda
    w_final = _fit_ridge(F_std, y_train, best_lambda)
    
    # Check if we should also try a pure least-squares fit (lam=0)
    # and compare
    if _time_left() > 5:
        w_lsq, *_ = np.linalg.lstsq(F_std, y_train, rcond=None)
        pred_lsq = F_std @ w_lsq
        mse_lsq = np.mean((pred_lsq - y_train) ** 2)
        
        pred_ridge = F_std @ w_final
        mse_ridge = np.mean((pred_ridge - y_train) ** 2)
        
        # Use the one with lower training MSE (data is noiseless)
        if mse_lsq < mse_ridge:
            w_final = w_lsq
    
    def predict(X):
        F = _build_basis(X)
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        F_std_pred = (F - feat_mean) / feat_std
        return F_std_pred @ w_final
    
    return predict
# EVOLVE-BLOCK-END
