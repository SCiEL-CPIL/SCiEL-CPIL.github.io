"""Symbolic regression over 10 hidden equations using Kernel Ridge Regression ensemble."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    """
    Symbolic regression using an ensemble of Kernel Ridge Regressors.
    Tries RBF, Polynomial, and Linear kernels with multiple hyperparameters,
    selects the best via cross-validation, and returns a predict callable.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # 20s safety margin before 300s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).flatten()
    n_samples, n_features = X.shape

    # Handle edge case: constant y
    var_y = np.var(y)
    if var_y < 1e-12:
        mean_y = np.mean(y)
        return lambda X_test: np.full(len(X_test), mean_y)

    # Standardize X for better kernel behavior
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_std = X_std  # keep for predict

    def standardize(X_in):
        return (X_in - X_mean) / X_std

    Xs = standardize(X)

    # Cross-validation split
    np.random.seed(42)
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    n_folds = 5
    fold_size = n_samples // n_folds
    folds = [idx[i*fold_size:(i+1)*fold_size] for i in range(n_folds)]
    folds[-1] = np.append(folds[-1], idx[(n_folds-1)*fold_size:])  # handle remainder

    def cv_score(kernel_type, **params):
        """Compute mean CV score (MSE/var_y) for a given kernel configuration."""
        total_mse = 0.0
        n_eval = 0
        for k in range(n_folds):
            if time.time() > deadline:
                return np.inf
            val_idx = folds[k]
            tr_idx = np.concatenate([folds[i] for i in range(n_folds) if i != k])
            
            X_tr_s = Xs[tr_idx]
            y_tr = y[tr_idx]
            X_va_s = Xs[val_idx]
            y_va = y[val_idx]
            
            try:
                pred = _krr_predict(X_tr_s, y_tr, X_va_s, kernel_type, **params)
                mse = np.mean((pred - y_va) ** 2)
                if np.isfinite(mse):
                    total_mse += mse
                    n_eval += 1
            except Exception:
                return np.inf
        if n_eval == 0:
            return np.inf
        return total_mse / n_eval / var_y

    def _krr_predict(X_tr, y_tr, X_test, kernel_type, **params):
        """Fit KRR on X_tr/y_tr and predict on X_test."""
        n_tr = len(X_tr)
        
        if kernel_type == 'rbf':
            gamma = params.get('gamma', 1.0)
            K_tr = np.exp(-gamma * cdist(X_tr, X_tr, 'sqeuclidean'))
            K_va = np.exp(-gamma * cdist(X_va, X_tr, 'sqeuclidean'))
            alpha = params.get('alpha', 1e-3)
            A = K_tr + alpha * np.eye(n_tr)
            w = np.linalg.solve(A, y_tr)
            return K_va @ w
        elif kernel_type == 'poly':
            degree = params.get('degree', 3)
            coef0 = params.get('coef0', 1.0)
            gamma = params.get('gamma', 1.0)
            K_tr = (gamma * X_tr @ X_tr.T + coef0) ** degree
            K_va = (gamma * X_va @ X_tr.T + coef0) ** degree
            alpha = params.get('alpha', 1e-3)
            A = K_tr + alpha * np.eye(n_tr)
            w = np.linalg.solve(A, y_tr)
            return K_va @ w
        elif kernel_type == 'linear':
            K_tr = X_tr @ X_tr.T
            K_va = X_va @ X_tr.T
            alpha = params.get('alpha', 1e-3)
            A = K_tr + alpha * np.eye(n_tr)
            w = np.linalg.solve(A, y_tr)
            return K_va @ w
        elif kernel_type == 'spline':
            gamma = params.get('gamma', 1.0)
            K_tr = np.power(1 + cdist(X_tr, X_tr, 'sqeuclidean') / (6.0 * gamma), -3.0)
            K_va = np.power(1 + cdist(X_va, X_tr, 'sqeuclidean') / (6.0 * gamma), -3.0)
            alpha = params.get('alpha', 1e-3)
            A = K_tr + alpha * np.eye(n_tr)
            w = np.linalg.solve(A, y_tr)
            return K_va @ w
        else:
            raise ValueError(f"Unknown kernel: {kernel_type}")

    # Define candidate kernel configurations
    candidates = []
    
    # RBF with various gammas
    for gamma in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        for alpha in [1e-4, 1e-3, 1e-2, 0.1]:
            candidates.append(('rbf', {'gamma': gamma, 'alpha': alpha}))
    
    # Polynomial kernels
    for degree in [2, 3, 4]:
        for gamma in [0.1, 0.5, 1.0]:
            for coef0 in [0.5, 1.0, 2.0]:
                for alpha in [1e-3, 1e-2, 0.1]:
                    candidates.append(('poly', {'degree': degree, 'gamma': gamma, 'coef0': coef0, 'alpha': alpha}))
    
    # Linear kernel
    for alpha in [1e-4, 1e-3, 1e-2, 0.1, 1.0]:
        candidates.append(('linear', {'alpha': alpha}))
    
    # Spline kernels
    for gamma in [0.5, 1.0, 2.0]:
        for alpha in [1e-3, 1e-2, 0.1]:
            candidates.append(('spline', {'gamma': gamma, 'alpha': alpha}))

    # Evaluate candidates via CV, keep track of best
    best_cv_score = np.inf
    best_config = None
    
    # Sort by a heuristic: try simpler/more likely configs first
    # RBF first (most flexible), then poly, then linear
    rbf_configs = [c for c in candidates if c[0] == 'rbf']
    poly_configs = [c for c in candidates if c[0] == 'poly']
    linear_configs = [c for c in candidates if c[0] == 'linear']
    spline_configs = [c for c in candidates if c[0] == 'spline']
    
    # Sort RBF by gamma (medium first) and alpha (smaller first)
    rbf_configs.sort(key=lambda c: (abs(np.log2(c[1]['gamma'])), np.log2(c[1]['alpha'])))
    
    ordered_candidates = rbf_configs + spline_configs + poly_configs + linear_configs
    
    for config in ordered_candidates:
        if time.time() > deadline:
            break
        kernel_type, params = config
        score = cv_score(kernel_type, **params)
        if np.isfinite(score) and score < best_cv_score:
            best_cv_score = score
            best_config = config
    
    # If no config was found (shouldn't happen), fall back to simple linear
    if best_config is None:
        best_config = ('linear', {'alpha': 1e-2})
    
    # Final fit on all data with best config
    kernel_type, params = best_config
    Xs_all = standardize(X)
    
    def predict(X_test):
        X_test_s = standardize(np.asarray(X_test, dtype=np.float64))
        n_tr = len(Xs_all)
        
        if kernel_type == 'rbf':
            gamma = params['gamma']
            K_test = np.exp(-gamma * cdist(X_test_s, Xs_all, 'sqeuclidean'))
            K_tr = np.exp(-gamma * cdist(Xs_all, Xs_all, 'sqeuclidean'))
            alpha = params['alpha']
            A = K_tr + alpha * np.eye(n_tr)
            w = np.linalg.solve(A, y)
            return K_test @ w
        elif kernel_type == 'poly':
            degree = params['degree']
            coef0 = params['coef0']
            gamma = params['gamma']
            K_test = (gamma * X_test_s @ Xs_all.T + coef0) ** degree
            K_tr = (gamma * Xs_all @ Xs_all.T + coef0) ** degree
            alpha = params['alpha']
            A = K_tr + alpha * np.eye(n_tr)
            w = np.linalg.solve(A, y)
            return K_test @ w
        elif kernel_type == 'linear':
            K_test = X_test_s @ Xs_all.T
            K_tr = Xs_all @ Xs_all.T
            alpha = params['alpha']
            A = K_tr + alpha * np.eye(n_tr)
            w = np.linalg.solve(A, y)
            return K_test @ w
        elif kernel_type == 'spline':
            gamma = params['gamma']
            K_test = np.power(1 + cdist(X_test_s, Xs_all, 'sqeuclidean') / (6.0 * gamma), -3.0)
            K_tr = np.power(1 + cdist(Xs_all, Xs_all, 'sqeuclidean') / (6.0 * gamma), -3.0)
            alpha = params['alpha']
            A = K_tr + alpha * np.eye(n_tr)
            w = np.linalg.solve(A, y)
            return K_test @ w
        else:
            return np.full(len(X_test), np.mean(y))
    
    return predict

# EVOLVE-BLOCK-END
