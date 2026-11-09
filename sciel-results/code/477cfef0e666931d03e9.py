"""Symbolic regression with GP + polynomial fallback and time budget."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import warnings


def _rbf_kernel(X1, X2, length_scale, variance):
    """Compute RBF kernel matrix between X1 and X2."""
    X1 = np.asarray(X1, dtype=np.float64)
    X2 = np.asarray(X2, dtype=np.float64)
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    # Compute squared distances
    sq1 = np.sum(X1 ** 2, axis=1)[:, np.newaxis]
    sq2 = np.sum(X2 ** 2, axis=1)[np.newaxis, :]
    sq_dist = sq1 + sq2 - 2.0 * X1 @ X2.T
    np.maximum(sq_dist, 0, out=sq_dist)
    K = variance * np.exp(-0.5 * sq_dist / (length_scale ** 2))
    return K


def _fit_gp(X_train, y_train, max_iter=50, time_budget=280.0):
    """Fit a Gaussian Process with RBF kernel via MLE."""
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Standardize inputs
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd
    
    # Initial hyperparameters (log scale for optimization)
    # length_scale, variance, noise
    init_params = np.array([np.log(1.0), np.log(1.0), np.log(1e-6)])
    
    def log_marginal_likelihood(params):
        log_ls, log_var, log_noise = params
        ls = np.exp(log_ls)
        var = np.exp(log_var)
        noise = np.exp(log_noise)
        
        K = _rbf_kernel(Xs, Xs, ls, var)
        K += noise * np.eye(n)
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -1e10
        
        # log |K| = 2 * sum(log(diag(L)))
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        # alpha = K^{-1} y
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        lml = -0.5 * y @ alpha - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)
        return -lml  # negative for minimization
    
    def predict_gp(X_test, log_ls, log_var, log_noise):
        ls = np.exp(log_ls)
        var = np.exp(log_var)
        noise = np.exp(log_noise)
        
        X_test_s = (X_test - mu) / sd
        K_ss = _rbf_kernel(X_test_s, Xs, ls, var)
        K = _rbf_kernel(Xs, Xs, ls, var)
        K += noise * np.eye(n)
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return np.full(X_test.shape[0], np.mean(y))
        
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        mean = K_ss @ alpha
        return mean
    
    t0 = time.time()
    best_params = init_params.copy()
    best_lml = -log_marginal_likelihood(init_params)
    
    # Optimize hyperparameters
    try:
        result = minimize(
            log_marginal_likelihood,
            init_params,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'maxfun': 200},
            bounds=[(-5, 5), (-5, 5), (-15, 0)]
        )
        if result.fun < log_marginal_likelihood(init_params):
            best_params = result.x
            best_lml = -result.fun
    except Exception:
        pass
    
    # Check time
    if time.time() - t0 > time_budget:
        return None, None
    
    log_ls, log_var, log_noise = best_params
    
    def predict(X_test):
        return predict_gp(X_test, log_ls, log_var, log_noise)
    
    return predict, best_lml


def _fit_polynomial(X_train, y_train, degree=3, ridge_lambdas=None):
    """Fit a polynomial regression with ridge penalty."""
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Standardize
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd
    
    def make_features(Xraw):
        Xr = np.asarray(Xraw, dtype=np.float64)
        Xrn = (Xr - mu) / sd
        cols = [np.ones(Xrn.shape[0])]
        # Linear terms
        for j in range(d):
            cols.append(Xrn[:, j])
        # Quadratic terms
        for i in range(d):
            for j in range(i, d):
                cols.append(Xrn[:, i] * Xrn[:, j])
        # Cubic terms (only if degree >= 3)
        if degree >= 3:
            for i in range(d):
                for j in range(d):
                    for k in range(j, d):
                        cols.append(Xrn[:, i] * Xrn[:, j] * Xrn[:, k])
        return np.column_stack(cols)
    
    F_train = make_features(Xs)
    
    if ridge_lambdas is None:
        ridge_lambdas = [0.0, 0.1, 1.0, 10.0, 100.0]
    
    best_w = None
    best_mse = np.inf
    best_lambda = 0.0
    
    for lam in ridge_lambdas:
        try:
            if lam == 0.0:
                w, *_ = np.linalg.lstsq(F_train, y, rcond=None)
            else:
                A = F_train.T @ F_train + lam * np.eye(F_train.shape[1])
                b = F_train.T @ y
                w = np.linalg.solve(A, b)
            
            # Evaluate on training data (proxy for test)
            pred = F_train @ w
            mse = np.mean((pred - y) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_w = w
                best_lambda = lam
        except Exception:
            continue
    
    if best_w is None:
        # Fallback to simple linear
        w, *_ = np.linalg.lstsq(F_train, y, rcond=None)
        best_w = w
    
    def predict(X_test):
        Xt = np.asarray(X_test, dtype=np.float64)
        Ft = make_features(Xt)
        return Ft @ best_w
    
    return predict


def fit(X_train, y_train):
    """Main fit function that tries GP first, falls back to polynomial."""
    t0 = time.time()
    DEADLINE = 325.0  # safety margin before 330s kill
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    # Try GP first
    try:
        gp_predict, lml = _fit_gp(X, y, max_iter=40, time_budget=250.0)
        if gp_predict is not None and time.time() - t0 < DEADLINE:
            # Validate GP prediction is finite
            test_pred = gp_predict(X)
            if np.all(np.isfinite(test_pred)):
                return gp_predict
    except Exception:
        pass
    
    # Fall back to polynomial
    try:
        poly_predict = _fit_polynomial(X, y, degree=3)
        test_pred = poly_predict(X)
        if np.all(np.isfinite(test_pred)):
            return poly_predict
    except Exception:
        pass
    
    # Final fallback: mean
    y_mean = np.mean(y)
    def predict(X_test):
        return np.full(X_test.shape[0], y_mean)
    return predict
# EVOLVE-BLOCK-END
