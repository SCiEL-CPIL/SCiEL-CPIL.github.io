"""Symbolic regression with kernel-based adaptive fit and CV model selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

_START_TIME = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 20.0


def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)


def _poly_features(X, degree):
    """Polynomial features for 1D input."""
    x = X[:, 0]
    cols = [np.ones(len(x))]
    for k in range(1, degree + 1):
        cols.append(x ** k)
    return np.column_stack(cols)


def _trig_features(X, n_harmonics=5):
    """Trigonometric features for 1D input."""
    x = X[:, 0]
    cols = [np.ones(len(x))]
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    return np.column_stack(cols)


def _mixed_features(X, poly_deg=3, n_harmonics=3):
    """Mixed polynomial and trigonometric features."""
    x = X[:, 0]
    cols = [np.ones(len(x))]
    for k in range(1, poly_deg + 1):
        cols.append(x ** k)
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    return np.column_stack(cols)


def _rational_features(X, num_deg=2, den_deg=1):
    """Rational function features."""
    x = X[:, 0]
    cols = [np.ones(len(x))]
    for k in range(1, num_deg + 1):
        cols.append(x ** k)
    denom = 1 + x ** 2
    for k in range(1, den_deg + 1):
        cols.append(x ** k / denom)
    return np.column_stack(cols)


def _exp_features(X):
    """Exponential and log features."""
    x = X[:, 0]
    cols = [np.ones(len(x)), x, x ** 2]
    x_clipped = np.clip(x, -10, 10)
    cols.append(np.exp(x_clipped))
    cols.append(np.exp(-x_clipped))
    x_safe = np.where(np.abs(x) > 1e-6, x, 1e-6)
    cols.append(np.log(np.abs(x_safe)))
    return np.column_stack(cols)


def _rich_features_1d(X, max_poly=10, max_trig=6):
    """Rich feature set combining poly, trig, exp, rational, sqrt."""
    x = X[:, 0]
    n = len(x)
    cols = [np.ones(n)]
    
    # Polynomial
    for k in range(1, max_poly + 1):
        cols.append(x ** k)
    
    # Trigonometric
    for k in range(1, max_trig + 1):
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    
    # Exponential (scaled)
    cols.append(np.exp(0.3 * x))
    cols.append(np.exp(-0.3 * x))
    cols.append(np.exp(0.7 * x))
    cols.append(np.exp(-0.7 * x))
    
    # Rational
    denom = 1 + x ** 2
    cols.append(x / denom)
    cols.append(x ** 2 / denom)
    cols.append(x ** 3 / denom)
    
    # Sqrt for positive domain
    pos_mask = x > 0
    if np.any(pos_mask):
        sqrt_x = np.zeros(n)
        sqrt_x[pos_mask] = np.sqrt(x[pos_mask])
        cols.append(sqrt_x)
        cols.append(x * sqrt_x)
    
    # Abs
    cols.append(np.abs(x))
    
    # Trig products
    cols.append(np.sin(x) * np.cos(x))
    cols.append(np.sin(x) ** 2)
    cols.append(np.cos(x) ** 2)
    
    return np.column_stack(cols)


def _rbf_kernel(X1, X2, gamma):
    """Compute RBF kernel matrix."""
    X1_sq = np.sum(X1 ** 2, axis=1)
    X2_sq = np.sum(X2 ** 2, axis=1)
    d2 = X1_sq[:, None] + X2_sq[None, :] - 2 * X1 @ X2.T
    d2 = np.maximum(d2, 0)
    return np.exp(-gamma * d2)


def _fit_rbf(X_train, y_train, gamma, alpha):
    """Fit RBF kernel ridge regression."""
    K = _rbf_kernel(X_train, X_train, gamma)
    n = K.shape[0]
    K_reg = K + alpha * np.eye(n)
    try:
        w = np.linalg.solve(K_reg, y_train)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(K_reg, y_train, rcond=None)[0]
    
    def predict(X):
        K_test = _rbf_kernel(X, X_train, gamma)
        return K_test @ w
    
    y_pred = K @ w
    mse = np.mean((y_pred - y_train) ** 2)
    return mse, predict


def _fit_linear_features(features_func, X_train, y_train, alpha=0.0):
    """Fit linear model on features with optional ridge regularization."""
    try:
        Xf = features_func(X_train)
        n, p = Xf.shape
        if alpha > 0:
            AtA = Xf.T @ Xf + alpha * np.eye(p)
            Aty = Xf.T @ y_train
            w = np.linalg.solve(AtA, Aty)
        else:
            w, *_ = np.linalg.lstsq(Xf, y_train, rcond=None)
        y_pred = Xf @ w
        mse = np.mean((y_pred - y_train) ** 2)
        
        def predict(X):
            Xf_test = features_func(X)
            return Xf_test @ w
        
        return mse, predict
    except Exception:
        return float('inf'), None


def _cross_validate(features_func, X_train, y_train, n_folds=5, alpha=0.0):
    """Simple k-fold CV to estimate generalization error."""
    n = len(X_train)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    fold_size = n // n_folds
    cv_errors = []
    
    for i in range(n_folds):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        
        try:
            Xf_tr = features_func(X_tr)
            Xf_val = features_func(X_val)
            if alpha > 0:
                p = Xf_tr.shape[1]
                AtA = Xf_tr.T @ Xf_tr + alpha * np.eye(p)
                Aty = Xf_tr.T @ y_tr
                w = np.linalg.solve(AtA, Aty)
            else:
                w, *_ = np.linalg.lstsq(Xf_tr, y_tr, rcond=None)
            y_pred = Xf_val @ w
            cv_errors.append(np.mean((y_pred - y_val) ** 2))
        except Exception:
            cv_errors.append(float('inf'))
    
    return np.mean(cv_errors)


def fit(X_train, y_train):
    global _START_TIME
    _START_TIME = time.time()
    
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    d = X_train.shape[1]
    
    if d != 1:
        # Fallback to simple polynomial for multi-dimensional
        def features(X):
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
        return lambda X: features(X) @ w
    
    # For 1D: try multiple model families with CV-based selection
    
    best_cv_mse = float('inf')
    best_predict = None
    best_train_mse = float('inf')
    best_predict_train = None
    
    # Define candidate models as (name, features_func, alpha)
    candidates = []
    
    # Polynomial models
    for deg in [3, 4, 5, 6, 7, 8, 10, 12]:
        for alpha in [0.0, 1e-4, 1e-2]:
            def make_poly(deg=deg, alpha=alpha):
                def feat(X):
                    return _poly_features(X, deg)
                feat._alpha = alpha
                return feat
            candidates.append(('poly', make_poly()))
    
    # Trigonometric models
    for n_h in [3, 5, 7, 10]:
        for alpha in [0.0, 1e-4]:
            def make_trig(n=n_h, alpha=alpha):
                def feat(X):
                    return _trig_features(X, n_harmonics=n)
                feat._alpha = alpha
                return feat
            candidates.append(('trig', make_trig()))
    
    # Mixed models
    for p, t in [(2, 2), (3, 3), (4, 4), (5, 5)]:
        for alpha in [0.0, 1e-4]:
            def make_mixed(p=p, t=t, alpha=alpha):
                def feat(X):
                    return _mixed_features(X, poly_deg=p, n_harmonics=t)
                feat._alpha = alpha
                return feat
            candidates.append(('mixed', make_mixed()))
    
    # Rational models
    for nd, dd in [(2, 1), (3, 1), (4, 2)]:
        for alpha in [0.0, 1e-4]:
            def make_rat(nd=nd, dd=dd, alpha=alpha):
                def feat(X):
                    return _rational_features(X, num_deg=nd, den_deg=dd)
                feat._alpha = alpha
                return feat
            candidates.append(('rational', make_rat()))
    
    # Exponential models
    for alpha in [0.0, 1e-4]:
        def make_exp(alpha=alpha):
            def feat(X):
                return _exp_features(X)
            feat._alpha = alpha
            return feat
        candidates.append(('exp', make_exp()))
    
    # Rich feature models
    for mp, mt in [(6, 4), (8, 5), (10, 6), (12, 8)]:
        for alpha in [0.0, 1e-4, 1e-2]:
            def make_rich(mp=mp, mt=mt, alpha=alpha):
                def feat(X):
                    return _rich_features_1d(X, max_poly=mp, max_trig=mt)
                feat._alpha = alpha
                return feat
            candidates.append(('rich', make_rich()))
    
    # RBF kernel models
    rbf_gammas = [0.1, 0.5, 1.0, 2.0, 5.0]
    rbf_alphas = [1e-3, 1e-2, 0.1, 1.0]
    
    # First pass: evaluate all candidates with CV
    for name, feat_func in candidates:
        if _time_left() < 10.0:
            break
        
        alpha = getattr(feat_func, '_alpha', 0.0)
        
        try:
            cv_mse = _cross_validate(feat_func, X_train, y_train, n_folds=5, alpha=alpha)
            if cv_mse < best_cv_mse:
                best_cv_mse = cv_mse
                # Fit on full data
                train_mse, predict = _fit_linear_features(feat_func, X_train, y_train, alpha)
                if predict is not None:
                    best_predict = predict
                    best_train_mse = train_mse
        except Exception:
            continue
    
    # Second pass: try RBF kernels
    for gamma in rbf_gammas:
        if _time_left() < 10.0:
            break
        for alpha in rbf_alphas:
            if _time_left() < 5.0:
                break
            try:
                # Quick CV for RBF
                n = len(X_train)
                indices = np.arange(n)
                np.random.seed(42)
                np.random.shuffle(indices)
                fold_size = n // 5
                cv_errors = []
                
                for i in range(5):
                    val_start = i * fold_size
                    val_end = val_start + fold_size if i < 4 else n
                    val_idx = indices[val_start:val_end]
                    train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
                    
                    X_tr = X_train[train_idx]
                    y_tr = y_train[train_idx]
                    X_val = X_train[val_idx]
                    y_val = y_train[val_idx]
                    
                    K_tr = _rbf_kernel(X_tr, X_tr, gamma)
                    K_reg = K_tr + alpha * np.eye(len(X_tr))
                    w = np.linalg.solve(K_reg, y_tr)
                    
                    K_val = _rbf_kernel(X_val, X_tr, gamma)
                    y_pred = K_val @ w
                    cv_errors.append(np.mean((y_pred - y_val) ** 2))
                
                cv_mse = np.mean(cv_errors)
                if cv_mse < best_cv_mse:
                    best_cv_mse = cv_mse
                    train_mse, predict = _fit_rbf(X_train, y_train, gamma, alpha)
                    if predict is not None:
                        best_predict = predict
                        best_train_mse = train_mse
            except Exception:
                continue
    
    # Fallback if nothing worked
    if best_predict is None:
        Xf = _poly_features(X_train, 3)
        w, *_ = np.linalg.lstsq(Xf, y_train, rcond=None)
        best_predict = lambda X, w=w: _poly_features(X, 3) @ w
    
    return best_predict
# EVOLVE-BLOCK-END
