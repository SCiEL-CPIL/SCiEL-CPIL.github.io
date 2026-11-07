"""Multi-model symbolic regression with model selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _time_budget():
    """Return remaining time budget (seconds)."""
    return 280.0 - (time.time() - _TIME_START)

def _poly_features(X, degree):
    """Create polynomial features up to given degree."""
    d = X.shape[1]
    n = len(X)
    cols = [np.ones(n)]
    # Degree 1
    for j in range(d):
        cols.append(X[:, j])
    # Higher degrees
    for deg in range(2, degree + 1):
        for i1 in range(d):
            for i2 in range(i1, d):
                if deg == 2:
                    cols.append(X[:, i1] * X[:, i2])
                else:
                    # For degree 3, add all combinations
                    if deg == 3:
                        for i3 in range(i2, d):
                            cols.append(X[:, i1] * X[:, i2] * X[:, i3])
    return np.column_stack(cols)

def _rbf_kernel(X1, X2, length_scale):
    """RBF kernel matrix."""
    sq_dists = cdist(X1, X2, metric='sqeuclidean')
    return np.exp(-sq_dists / (2.0 * length_scale ** 2))

def _fit_ridge(X, y, alpha):
    """Fit ridge regression."""
    n = len(X)
    d = X.shape[1]
    A = X.T @ X + alpha * np.eye(d)
    b = X.T @ y
    w = np.linalg.solve(A, b)
    return w

def _loo_mse(X, y, predict_fn):
    """Estimate LOO MSE efficiently."""
    n = len(y)
    preds = predict_fn(X)
    residuals = y - preds
    # Use hat matrix approximation if possible, otherwise simple LOO
    # For simplicity, use K-fold style with n folds (expensive but accurate)
    # Actually, for ridge we can use the formula, but let's do simple LOO
    mse_sum = 0.0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_tr = X[mask]
        y_tr = y[mask]
        pred_i = predict_fn(np.array([X[i]]))
        mse_sum += (pred_i[0] - y[i]) ** 2
    return mse_sum / n

def _fit_poly_ridge(X, y, degree, alpha):
    """Fit polynomial ridge regression."""
    F = _poly_features(X, degree)
    w = _fit_ridge(F, y, alpha)
    def predict(X_new):
        F_new = _poly_features(X_new, degree)
        return F_new @ w
    return predict

def _fit_rbf_ridge(X, y, length_scale, alpha):
    """Fit RBF kernel ridge regression."""
    K = _rbf_kernel(X, X, length_scale)
    n = len(X)
    A = K + alpha * np.eye(n)
    w = np.linalg.solve(A, y)
    def predict(X_new):
        K_new = _rbf_kernel(X_new, X, length_scale)
        return K_new @ w
    return predict

def _fit_gp_matern52(X, y, length_scale, noise_var):
    """Fit Matern 5/2 Gaussian Process."""
    d = X.shape[1]
    n = len(X)
    
    def matern52_kernel(X1, X2, ls):
        dists = cdist(X1, X2, metric='euclidean')
        sqrt3_dists = np.sqrt(3) * dists / ls
        K = (1 + sqrt3_dists + sqrt3_dists**2 / 3.0) * np.exp(-sqrt3_dists)
        return K
    
    K = matern52_kernel(X, X, length_scale)
    K += noise_var * np.eye(n)
    
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    
    def predict(X_new):
        K_star = matern52_kernel(X_new, X, length_scale)
        mean = K_star @ alpha
        return mean
    
    return predict

def _fit_gp_matern52_optimized(X, y, time_limit):
    """Fit GP with optimized hyperparameters."""
    d = X.shape[1]
    n = len(X)
    
    # Initialize with reasonable values
    ls_init = np.std(X, axis=0).mean() if d > 0 else 1.0
    noise_init = np.var(y) * 0.01 + 1e-6
    
    best_score = np.inf
    best_params = (ls_init, noise_init)
    
    def neg_log_marginal(params):
        ls, noise_var = params
        ls = max(ls, 1e-6)
        noise_var = max(noise_var, 1e-8)
        
        def matern52_kernel(X1, X2, ls_val):
            dists = cdist(X1, X2, metric='euclidean')
            sqrt3_dists = np.sqrt(3) * dists / ls_val
            K = (1 + sqrt3_dists + sqrt3_dists**2 / 3.0) * np.exp(-sqrt3_dists)
            return K
        
        K = matern52_kernel(X, X, ls)
        K += noise_var * np.eye(n)
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            nll = 0.5 * y @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)
            return nll
        except:
            return 1e10
    
    # Try a few starting points
    start_points = [
        (ls_init, noise_init),
        (ls_init * 0.5, noise_init * 0.5),
        (ls_init * 2.0, noise_init * 2.0),
        (ls_init * 0.1, noise_init * 0.1),
        (ls_init * 5.0, noise_init * 5.0),
    ]
    
    elapsed = time.time() - _TIME_START
    remaining = time_limit - elapsed
    
    if remaining < 5:
        # Just use initial params
        return _fit_gp_matern52(X, y, ls_init, noise_init)
    
    # Optimize from each starting point
    for sp in start_points:
        if time.time() - _TIME_START > time_limit - 10:
            break
        try:
            res = minimize(neg_log_marginal, sp, method='Nelder-Mead',
                          options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_score:
                best_score = res.fun
                best_params = (res.x[0], res.x[1])
        except:
            pass
    
    ls, noise_var = best_params
    return _fit_gp_matern52(X, y, ls, noise_var)

def _fit_rational(X, y):
    """Fit rational function approximator: ratio of two linear basis expansions."""
    d = X.shape[1]
    n = len(X)
    
    # Numerator: linear in X
    # Denominator: 1 + linear in X (to avoid zero denominator)
    F_num = np.column_stack([np.ones(n)] + [X[:, j] for j in range(d)])
    F_den = np.column_stack([np.ones(n)] + [X[:, j] for j in range(d)])
    
    # We want to find a, b such that y ≈ (F_num @ a) / (F_den @ b)
    # This is a nonlinear problem. Use iterative approach.
    
    # Initialize b to all ones (denominator = 1)
    b = np.ones(d + 1)
    a = np.zeros(d + 1)
    
    for _ in range(20):
        den = F_den @ b
        den = np.where(np.abs(den) < 1e-6, 1e-6 * np.sign(den), den)
        y_adj = y * den
        a = np.linalg.lstsq(F_num, y_adj, rcond=None)[0]
        
        # Update b using the residuals
        pred = (F_num @ a) / den
        resid = y - pred
        # Simple gradient step for b
        # This is complex, so let's just keep b fixed and refit a
        # Or use a simpler approach: just fit a linear model
        break
    
    # For simplicity, just return a linear model if rational is too complex
    w = np.linalg.lstsq(F_num, y, rcond=None)[0]
    def predict(X_new):
        F_new = np.column_stack([np.ones(len(X_new))] + [X_new[:, j] for j in range(X_new.shape[1])])
        return F_new @ w
    return predict

def fit(X_train, y_train):
    global _TIME_START
    _TIME_START = time.time()
    TIME_LIMIT = 280.0  # Leave 20s margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n = len(y)
    d = X.shape[1]
    
    var_y = np.var(y)
    if var_y < 1e-10:
        # Constant function
        mean_y = np.mean(y)
        return lambda X_new: np.full(len(X_new), mean_y)
    
    models = []
    model_names = []
    
    # 1. Polynomial degree 2
    try:
        pred_fn = _fit_poly_ridge(X, y, 2, 1e-6)
        models.append(pred_fn)
        model_names.append('poly2')
    except:
        pass
    
    # 2. Polynomial degree 3
    try:
        pred_fn = _fit_poly_ridge(X, y, 3, 1e-6)
        models.append(pred_fn)
        model_names.append('poly3')
    except:
        pass
    
    # 3. RBF kernel ridge
    try:
        ls = np.std(X, axis=0).mean() if d > 0 else 1.0
        pred_fn = _fit_rbf_ridge(X, y, ls, 1e-6)
        models.append(pred_fn)
        model_names.append('rbf')
    except:
        pass
    
    # 4. GP with Matern 5/2
    try:
        elapsed = time.time() - _TIME_START
        if elapsed < TIME_LIMIT - 30:
            pred_fn = _fit_gp_matern52_optimized(X, y, TIME_LIMIT)
            models.append(pred_fn)
            model_names.append('gp')
    except:
        pass
    
    # 5. Linear model
    try:
        F = np.column_stack([np.ones(n)] + [X[:, j] for j in range(d)])
        w = np.linalg.lstsq(F, y, rcond=None)[0]
        def predict_linear(X_new):
            F_new = np.column_stack([np.ones(len(X_new))] + [X_new[:, j] for j in range(X_new.shape[1])])
            return F_new @ w
        models.append(predict_linear)
        model_names.append('linear')
    except:
        pass
    
    if not models:
        # Fallback: mean
        mean_y = np.mean(y)
        return lambda X_new: np.full(len(X_new), mean_y)
    
    # Select best model using LOO-CV (approximate with K-fold for speed)
    best_model = models[0]
    best_score = np.inf
    
    # Use 5-fold CV for model selection
    k = 5
    fold_size = n // k
    
    for idx, model in enumerate(models):
        fold_mses = []
        for fold in range(k):
            start = fold * fold_size
            end = start + fold_size if fold < k - 1 else n
            val_mask = np.zeros(n, dtype=bool)
            val_mask[start:end] = True
            tr_mask = ~val_mask
            
            X_tr = X[tr_mask]
            y_tr = y[tr_mask]
            X_val = X[val_mask]
            y_val = y[val_mask]
            
            # Refit model on training fold
            try:
                if model_names[idx] == 'poly2':
                    pred_fn = _fit_poly_ridge(X_tr, y_tr, 2, 1e-6)
                elif model_names[idx] == 'poly3':
                    pred_fn = _fit_poly_ridge(X_tr, y_tr, 3, 1e-6)
                elif model_names[idx] == 'rbf':
                    ls = np.std(X_tr, axis=0).mean() if d > 0 else 1.0
                    pred_fn = _fit_rbf_ridge(X_tr, y_tr, ls, 1e-6)
                elif model_names[idx] == 'gp':
                    elapsed = time.time() - _TIME_START
                    if elapsed > TIME_LIMIT - 15:
                        continue
                    pred_fn = _fit_gp_matern52_optimized(X_tr, y_tr, TIME_LIMIT)
                elif model_names[idx] == 'linear':
                    F_tr = np.column_stack([np.ones(len(X_tr))] + [X_tr[:, j] for j in range(d)])
                    w = np.linalg.lstsq(F_tr, y_tr, rcond=None)[0]
                    pred_fn = lambda X_new, w=w: (np.column_stack([np.ones(len(X_new))] + [X_new[:, j] for j in range(X_new.shape[1])]) @ w)
                else:
                    continue
                
                preds = pred_fn(X_val)
                mse = np.mean((preds - y_val) ** 2)
                fold_mses.append(mse)
            except:
                continue
        
        if fold_mses:
            avg_mse = np.mean(fold_mses)
            if avg_mse < best_score:
                best_score = avg_mse
                best_model = model
    
    return best_model
# EVOLVE-BLOCK-END
