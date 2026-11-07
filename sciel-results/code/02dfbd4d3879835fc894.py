"""Symbolic regression over 10 hidden equations: Expanded feature search + RBF with CV."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy.linalg

_START = time.time()
_DEADLINE = 300.0  # safety margin vs 330s kill

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _make_predict_rbf(X_train, y_train, gamma, alpha):
    """Create a predict function for RBF Kernel Ridge Regression."""
    diff = X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=-1)
    K = np.exp(-gamma * sq_dists)
    K_reg = K + alpha * np.eye(len(X_train))
    try:
        w = scipy.linalg.solve(K_reg, y_train)
    except Exception:
        w = np.linalg.pinv(K_reg) @ y_train
        
    def predict(X):
        diff_test = X[:, np.newaxis, :] - X_train[np.newaxis, :, :]
        sq_dists_test = np.sum(diff_test**2, axis=-1)
        K_test = np.exp(-gamma * sq_dists_test)
        return K_test @ w
        
    return predict

def _make_predict_linear(basis_func, w):
    """Create a predict function for linear combination of basis features."""
    def predict(X):
        A = basis_func(X)
        return A @ w
    return predict

def _try_exact_fit(basis_func, X, y, tol=1e-6):
    """Try to fit y using linear combination of basis_func(X)."""
    try:
        A = basis_func(X)
        if not np.all(np.isfinite(A)):
            return None
        w, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ w
        if not np.all(np.isfinite(pred)):
            return None
        mse = np.mean((pred - y)**2)
        scale = max(1.0, np.max(np.abs(y)))
        if mse <= (tol * scale)**2:
            return w, mse
    except Exception:
        pass
    return None

# --- Basis Function Generators ---

def _basis_poly2(X):
    d = X.shape[1]
    cols = [np.ones(len(X))]
    for j in range(d):
        cols.append(X[:, j])
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    return np.column_stack(cols)

def _basis_poly3(X):
    d = X.shape[1]
    cols = [np.ones(len(X))]
    for j in range(d):
        cols.append(X[:, j])
        cols.append(X[:, j] ** 2)
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    for i in range(d):
        cols.append(X[:, i] ** 3)
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append(X[:, i] * X[:, j] * X[:, k])
    return np.column_stack(cols)

def _basis_poly4(X):
    d = X.shape[1]
    cols = [np.ones(len(X))]
    for j in range(d):
        cols.append(X[:, j])
        cols.append(X[:, j] ** 2)
        cols.append(X[:, j] ** 3)
        cols.append(X[:, j] ** 4)
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
            cols.append(X[:, i] ** 2 * X[:, j])
            cols.append(X[:, i] * X[:, j] ** 2)
            cols.append(X[:, i] ** 2 * X[:, j] ** 2)
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append(X[:, i] * X[:, j] * X[:, k])
    return np.column_stack(cols)

def _basis_rich(X):
    """Includes polynomials, transcendentals, and common functions."""
    d = X.shape[1]
    cols = [np.ones(len(X))]
    
    # Polynomials up to degree 2
    for j in range(d):
        cols.append(X[:, j])
        cols.append(X[:, j] ** 2)
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
            
    # Transcendentals and common functions on individual variables
    for j in range(d):
        x = X[:, j]
        cols.append(np.sin(x))
        cols.append(np.cos(x))
        cols.append(np.tanh(x))
        cols.append(np.arctan(x))
        cols.append(np.log(x))
        cols.append(np.sqrt(x))
        cols.append(np.abs(x))
        cols.append(x ** 0.5)
        cols.append(np.exp(-x))  # decaying exp
        cols.append(np.sin(x * x))  # sin of square
        cols.append(np.cos(x * x))  # cos of square
        
    # Some key interactions
    if d >= 2:
        for i in range(d):
            for j in range(i+1, d):
                cols.append(np.sin(X[:, i]) * X[:, j])
                cols.append(np.tanh(X[:, i]) * X[:, j])
                cols.append(np.log(X[:, i]) * X[:, j])
                
    return np.column_stack(cols)

def _basis_trig_poly(X):
    """Polynomials and trigonometric functions with more interactions."""
    d = X.shape[1]
    cols = [np.ones(len(X))]
    for j in range(d):
        cols.append(X[:, j])
        cols.append(np.sin(X[:, j]))
        cols.append(np.cos(X[:, j]))
        cols.append(np.tanh(X[:, j]))
        cols.append(np.arctan(X[:, j]))
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
            cols.append(np.sin(X[:, i]) * np.cos(X[:, j]))
            cols.append(np.sin(X[:, i] + X[:, j]))
            cols.append(np.cos(X[:, i] + X[:, j]))
    return np.column_stack(cols)

def _basis_comprehensive(X):
    """Very comprehensive basis for complex functions."""
    d = X.shape[1]
    cols = [np.ones(len(X))]
    
    # All polynomial terms up to degree 3
    for j in range(d):
        cols.append(X[:, j])
        cols.append(X[:, j] ** 2)
        cols.append(X[:, j] ** 3)
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
            cols.append(X[:, i] ** 2 * X[:, j])
            cols.append(X[:, i] * X[:, j] ** 2)
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append(X[:, i] * X[:, j] * X[:, k])
    
    # Transcendentals
    for j in range(d):
        x = X[:, j]
        cols.append(np.sin(x))
        cols.append(np.cos(x))
        cols.append(np.tanh(x))
        cols.append(np.arctan(x))
        cols.append(np.log(x))
        cols.append(np.sqrt(x))
        cols.append(np.exp(-x))
        cols.append(np.sin(x**2))
        cols.append(np.cos(x**2))
        cols.append(np.tanh(x**2))
        
    # Interactions with transcendentals
    if d >= 2:
        for i in range(d):
            for j in range(i+1, min(i+2, d)):  # limit to nearby interactions
                cols.append(np.sin(X[:, i]) * X[:, j])
                cols.append(np.tanh(X[:, i]) * X[:, j])
                cols.append(np.log(X[:, i]) * X[:, j])
                cols.append(X[:, i] * np.sin(X[:, j]))
                cols.append(X[:, i] * np.tanh(X[:, j]))
                
    return np.column_stack(cols)

_CANDIDATE_BASES = [
    _basis_poly2,
    _basis_poly3,
    _basis_poly4,
    _basis_rich,
    _basis_trig_poly,
    _basis_comprehensive
]

def _cv_rbf_score(X, y, gamma, alpha, n_folds=5):
    """Cross-validate RBF parameters."""
    n = len(y)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    cv_errors = []
    
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        if len(train_idx) < 10:
            continue
            
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        try:
            pred = _make_predict_rbf(X_train_fold, y_train_fold, gamma, alpha)(X_val_fold)
            err = np.mean((pred - y_val_fold)**2)
            if np.isfinite(err):
                cv_errors.append(err)
        except Exception:
            continue
    
    if cv_errors:
        return np.mean(cv_errors)
    return float('inf')

def fit(X_train, y_train):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = y.reshape(-1)
    
    n_samples = len(X)
    d = X.shape[1]
    
    # 1. Try Exact Fit with various bases
    best_exact_predict = None
    best_exact_err = float('inf')
    
    for basis_func in _CANDIDATE_BASES:
        if _remaining() < 15.0:
            break
        try:
            result = _try_exact_fit(basis_func, X, y)
            if result is not None:
                w, err = result
                if err < best_exact_err:
                    best_exact_err = err
                    best_exact_predict = _make_predict_linear(basis_func, w)
        except Exception:
            continue
            
    if best_exact_predict is not None:
        return best_exact_predict
        
    # 2. Fallback: RBF Kernel Ridge Regression with CV
    best_rbf_predict = None
    best_rbf_cv_err = float('inf')
    
    gammas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    alphas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    
    for gamma in gammas:
        if _remaining() < 10.0:
            break
        for alpha in alphas:
            if _remaining() < 5.0:
                break
            try:
                cv_err = _cv_rbf_score(X, y, gamma, alpha)
                if cv_err < best_rbf_cv_err:
                    best_rbf_cv_err = cv_err
                    best_rbf_predict = _make_predict_rbf(X, y, gamma, alpha)
            except Exception:
                continue
                
    if best_rbf_predict is not None:
        return best_rbf_predict
        
    # 3. Ultimate Fallback: Ridge regression on poly2
    try:
        A = _basis_poly2(X)
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        return _make_predict_linear(_basis_poly2, w)
    except Exception:
        mean_y = np.mean(y)
        return lambda X: np.full(len(X), mean_y)

# EVOLVE-BLOCK-END
