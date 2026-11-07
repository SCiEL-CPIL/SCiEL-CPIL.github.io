"""Symbolic regression with kernel ridge regression and adaptive hyperparameters."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import RBFInterpolator
from itertools import product

# Global time tracker
_total_start = None
_time_limit = 290.0  # Safety margin before 330s kill

def _elapsed():
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    return time.time() - _total_start

def _remaining():
    return _time_limit - _elapsed()

def fit(X_train, y_train):
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    
    n, d = X_train.shape
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    
    # Store original stats for prediction
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Strategy 1: Kernel Ridge Regression with RBF kernel and adaptive hyperparameters
    # We'll try multiple values of epsilon (inverse length scale) and alpha (regularization)
    
    best_krr_score = np.inf
    best_krr_params = None  # (epsilon, alpha)
    best_krr_weights = None
    
    # Normalize inputs to [0, 1] for better kernel behavior
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Grid search over epsilon and alpha - expanded for better coverage
    epsilon_values = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    alpha_values = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    
    sq_norms = np.sum(X_norm ** 2, axis=1)
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2 * X_norm @ X_norm.T
    np.maximum(sq_dists, 0, out=sq_dists)
    # Precompute squared distances once
    sq_dists_base = sq_dists.copy()
    
    for eps in epsilon_values:
        if _remaining() < 1.0:
            break
        # Compute RBF kernel matrix using precomputed distances
        K = np.exp(-eps * sq_dists_base)
        
        for alpha in alpha_values:
            if _remaining() < 1.0:
                break
            # Solve (K + alpha*I) w = y
            try:
                w = np.linalg.solve(K + alpha * np.eye(n), y_norm)
                pred = K @ w
                mse = np.mean((pred - y_norm) ** 2)
                if mse < best_krr_score:
                    best_krr_score = mse
                    best_krr_params = (eps, alpha)
                    best_krr_weights = w
            except np.linalg.LinAlgError:
                pass
    
    # Strategy 2: Polynomial kernel ridge regression - expanded
    best_poly_score = np.inf
    best_poly_params = None  # (degree, gamma, alpha)
    best_poly_weights = None
    
    degree_values = [2, 3, 4]
    gamma_values = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    
    for deg in degree_values:
        if _remaining() < 1.0:
            break
        for gamma in gamma_values:
            if _remaining() < 1.0:
                break
            # Polynomial kernel: K[i,j] = (gamma * x_i . x_j + 1)^deg
            K_poly = (gamma * X_norm @ X_norm.T + 1) ** deg
            
            for alpha in [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0]:
                if _remaining() < 1.0:
                    break
                try:
                    w = np.linalg.solve(K_poly + alpha * np.eye(n), y_norm)
                    pred = K_poly @ w
                    mse = np.mean((pred - y_norm) ** 2)
                    if mse < best_poly_score:
                        best_poly_score = mse
                        best_poly_params = (deg, gamma, alpha)
                        best_poly_weights = w
                except np.linalg.LinAlgError:
                    pass
    
    # Strategy 3: Rich feature expansion + Ridge regression
    def build_features(X):
        n = X.shape[0]
        d = X.shape[1]
        cols = [np.ones(n)]
        
        # Linear terms
        for j in range(d):
            cols.append(X[:, j])
        
        # Quadratic and cubic terms
        for deg in range(2, 5):
            monomials = []
            def _gen(idx, remaining, current):
                if idx == d - 1:
                    monomials.append(current + [remaining])
                    return
                for k in range(remaining + 1):
                    _gen(idx + 1, remaining - k, current + [k])
            _gen(0, deg, [])
            for exps in monomials:
                term = np.ones(n)
                for j in range(d):
                    if exps[j] > 0:
                        term = term * np.power(X[:, j], exps[j])
                cols.append(term)
        
        # Trigonometric terms
        for j in range(d):
            cols.append(np.sin(X[:, j]))
            cols.append(np.cos(X[:, j]))
            cols.append(np.sin(2 * X[:, j]))
            cols.append(np.cos(2 * X[:, j]))
        
        # Sin/cos of pairs
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(np.sin(X[:, i] + X[:, j]))
                cols.append(np.cos(X[:, i] + X[:, j]))
                cols.append(np.sin(X[:, i] - X[:, j]))
                cols.append(np.cos(X[:, i] - X[:, j]))
                cols.append(np.sin(X[:, i] * X[:, j]))
                cols.append(np.cos(X[:, i] * X[:, j]))
        
        # Exponential and log terms (scaled)
        for j in range(d):
            cols.append(np.exp(X[:, j] / 5.0))
            cols.append(np.exp(-X[:, j] / 5.0))
            cols.append(np.log(X[:, j] + 1.0))
            cols.append(np.log(X[:, j] + 0.1))
        
        # Products
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X[:, i] * X[:, j])
                cols.append(X[:, i] * X[:, j] ** 2)
                cols.append(X[:, i] ** 2 * X[:, j])
        
        # Ratios (safe)
        for i in range(d):
            for j in range(i + 1, d):
                denom = X[:, j] + 1e-10
                cols.append(X[:, i] / denom)
                denom2 = X[:, i] + 1e-10
                cols.append(X[:, j] / denom2)
        
        # Sum and diff
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X[:, i] + X[:, j])
                cols.append(X[:, i] - X[:, j])
        
        return np.column_stack(cols)
    
    F = build_features(X)
    n_features = F.shape[1]
    
    # Normalize features
    F_mean = F.mean(axis=0)
    F_std = F.std(axis=0)
    F_std[F_std < 1e-10] = 1.0
    F_norm = (F - F_mean) / F_std
    
    # Ridge regression with optimal lambda - expanded
    best_ridge_score = np.inf
    best_ridge_w = None
    best_ridge_lam = None
    
    lam_values = np.logspace(-6, 6, 25)
    
    for lam in lam_values:
        if _remaining() < 1.0:
            break
        A = F_norm.T @ F_norm + lam * np.eye(n_features)
        b = F_norm.T @ y_norm
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
        pred = F_norm @ w
        mse = np.mean((pred - y_norm) ** 2)
        if mse < best_ridge_score:
            best_ridge_score = mse
            best_ridge_w = w
            best_ridge_lam = lam
    
    # Strategy 4: For 1D, try specific nonlinear forms
    nonlinear_score = np.inf
    nonlinear_type = None
    nonlinear_params = None
    
    if d == 1:
        x = X[:, 0]
        
        # Power law: a * x^p + b
        def obj_power(params):
            if _remaining() < 1.0:
                return np.inf
            a, p, b = params
            try:
                pred = a * np.power(x, p) + b
                return np.mean((pred - y) ** 2)
            except:
                return np.inf
        
        try:
            res = minimize(obj_power, [1.0, 1.0, 0.0], method='Nelder-Mead',
                          options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500})
            if res.fun < nonlinear_score:
                nonlinear_score = res.fun
                nonlinear_type = 'power'
                nonlinear_params = res.x
        except:
            pass
        
        # Exponential: a * exp(p * x) + b
        def obj_exp(params):
            if _remaining() < 1.0:
                return np.inf
            a, p, b = params
            try:
                pred = a * np.exp(p * x) + b
                return np.mean((pred - y) ** 2)
            except:
                return np.inf
        
        try:
            res = minimize(obj_exp, [1.0, 0.1, 0.0], method='Nelder-Mead',
                          options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500})
            if res.fun < nonlinear_score:
                nonlinear_score = res.fun
                nonlinear_type = 'exp'
                nonlinear_params = res.x
        except:
            pass
        
        # Log: a * log(x + c) + b
        def obj_log(params):
            if _remaining() < 1.0:
                return np.inf
            a, c, b = params
            try:
                pred = a * np.log(x + c) + b
                return np.mean((pred - y) ** 2)
            except:
                return np.inf
        
        try:
            res = minimize(obj_log, [1.0, 1.0, 0.0], method='Nelder-Mead',
                          options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500})
            if res.fun < nonlinear_score:
                nonlinear_score = res.fun
                nonlinear_type = 'log'
                nonlinear_params = res.x
        except:
            pass
        
        # Sine: a * sin(p * x + phi) + b
        def obj_sin(params):
            if _remaining() < 1.0:
                return np.inf
            a, p, phi, b = params
            try:
                pred = a * np.sin(p * x + phi) + b
                return np.mean((pred - y) ** 2)
            except:
                return np.inf
        
        try:
            res = minimize(obj_sin, [1.0, 1.0, 0.0, 0.0], method='Nelder-Mead',
                          options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500})
            if res.fun < nonlinear_score:
                nonlinear_score = res.fun
                nonlinear_type = 'sin'
                nonlinear_params = res.x
        except:
            pass
        
        # Sqrt: a * sqrt(x + c) + b
        def obj_sqrt(params):
            if _remaining() < 1.0:
                return np.inf
            a, c, b = params
            try:
                pred = a * np.sqrt(x + c) + b
                return np.mean((pred - y) ** 2)
            except:
                return np.inf
        
        try:
            res = minimize(obj_sqrt, [1.0, 0.0, 0.0], method='Nelder-Mead',
                          options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500})
            if res.fun < nonlinear_score:
                nonlinear_score = res.fun
                nonlinear_type = 'sqrt'
                nonlinear_params = res.x
        except:
            pass
        
        # Rational: (a*x + b) / (c*x + d)
        def obj_rational(params):
            if _remaining() < 1.0:
                return np.inf
            a, b, c, d = params
            try:
                pred = (a * x + b) / (c * x + d + 1e-10)
                if not np.all(np.isfinite(pred)):
                    return np.inf
                return np.mean((pred - y) ** 2)
            except:
                return np.inf
        
        try:
            res = minimize(obj_rational, [1.0, 0.0, 0.1, 1.0], method='Nelder-Mead',
                          options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500})
            if res.fun < nonlinear_score:
                nonlinear_score = res.fun
                nonlinear_type = 'rational'
                nonlinear_params = res.x
        except:
            pass
    
    # Convert all scores to normalized MSE
    krr_score_norm = best_krr_score
    poly_score_norm = best_poly_score
    ridge_score_norm = best_ridge_score
    nonlinear_score_norm = nonlinear_score / (y_std ** 2 + 1e-10) if d == 1 else np.inf
    
    # Select best strategy
    candidates = []
    if best_krr_params is not None:
        candidates.append(('krr', krr_score_norm, best_krr_params, best_krr_weights))
    if best_poly_params is not None:
        candidates.append(('poly', poly_score_norm, best_poly_params, best_poly_weights))
    candidates.append(('ridge', ridge_score_norm, best_ridge_lam, best_ridge_w))
    if d == 1 and nonlinear_type is not None:
        candidates.append(('nonlinear', nonlinear_score_norm, nonlinear_params, nonlinear_type))
    
    best_strategy = min(candidates, key=lambda x: x[1])
    strategy_name, strategy_score, strat_params, extra = best_strategy
    
    # If we have multiple good strategies, create an ensemble
    # Sort by score and take top 2-3 if they're close
    candidates_sorted = sorted(candidates, key=lambda x: x[1])
    use_ensemble = False
    ensemble_strategies = []
    if len(candidates_sorted) >= 2:
        best_score = candidates_sorted[0][1]
        for c in candidates_sorted[:3]:
            if c[1] < best_score * 1.5:  # Within 50% of best
                ensemble_strategies.append(c)
        if len(ensemble_strategies) >= 2:
            use_ensemble = True
    
    # Build prediction function
    def _predict_krr(X_test, eps, w):
        X_t = X_test.astype(np.float64)
        X_t_norm = (X_t - X_min) / X_range
        sq_norms_test = np.sum(X_t_norm ** 2, axis=1)
        sq_norms_train = np.sum(X_norm ** 2, axis=1)
        sq_dists = sq_norms_test[:, None] + sq_norms_train[None, :] - 2 * X_t_norm @ X_norm.T
        np.maximum(sq_dists, 0, out=sq_dists)
        K_test = np.exp(-eps * sq_dists)
        y_pred_norm = K_test @ w
        return y_pred_norm * y_std + y_mean
    
    def _predict_poly(X_test, deg, gamma, w):
        X_t = X_test.astype(np.float64)
        X_t_norm = (X_t - X_min) / X_range
        K_test = (gamma * X_t_norm @ X_norm.T + 1) ** deg
        y_pred_norm = K_test @ w
        return y_pred_norm * y_std + y_mean
    
    def _predict_ridge(X_test, w):
        X_t = X_test.astype(np.float64)
        F_t = build_features(X_t)
        F_t_norm = (F_t - F_mean) / F_std
        y_pred_norm = F_t_norm @ w
        return y_pred_norm * y_std + y_mean
    
    def _predict_nonlinear(X_test, ntype, params):
        X_t = X_test.astype(np.float64)
        x = X_t[:, 0]
        if ntype == 'power':
            a, p, b = params
            return a * np.power(x, p) + b
        elif ntype == 'exp':
            a, p, b = params
            return a * np.exp(p * x) + b
        elif ntype == 'log':
            a, c, b = params
            return a * np.log(x + c) + b
        elif ntype == 'sin':
            a, p, phi, b = params
            return a * np.sin(p * x + phi) + b
        elif ntype == 'sqrt':
            a, c, b = params
            return a * np.sqrt(x + c) + b
        elif ntype == 'rational':
            a, b, c, d = params
            return (a * x + b) / (c * x + d + 1e-10)
        else:
            F_t = build_features(X_t)
            F_t_norm = (F_t - F_mean) / F_std
            y_pred_norm = F_t_norm @ best_ridge_w
            return y_pred_norm * y_std + y_mean
    
    if use_ensemble and len(ensemble_strategies) >= 2:
        # Weighted ensemble based on inverse score
        weights = []
        for c in ensemble_strategies:
            w_val = 1.0 / (c[1] + 1e-10)
            weights.append(w_val)
        total_w = sum(weights)
        weights = [w / total_w for w in weights]
        
        def predict(X_test):
            preds = []
            for w_val, c in zip(weights, ensemble_strategies):
                sname = c[0]
                if sname == 'krr':
                    eps, alpha = c[2]
                    preds.append(w_val * _predict_krr(X_test, eps, c[3]))
                elif sname == 'poly':
                    deg, gamma, alpha = c[2]
                    preds.append(w_val * _predict_poly(X_test, deg, gamma, c[3]))
                elif sname == 'ridge':
                    preds.append(w_val * _predict_ridge(X_test, c[3]))
                elif sname == 'nonlinear':
                    preds.append(w_val * _predict_nonlinear(X_test, c[3], c[2]))
            return sum(preds)
    
    elif strategy_name == 'krr':
        eps, alpha = strat_params
        w = extra
        def predict(X_test):
            return _predict_krr(X_test, eps, w)
    
    elif strategy_name == 'poly':
        deg, gamma, alpha = strat_params
        w = extra
        def predict(X_test):
            return _predict_poly(X_test, deg, gamma, w)
    
    elif strategy_name == 'ridge':
        lam = strat_params
        w = extra
        def predict(X_test):
            return _predict_ridge(X_test, w)
    
    elif strategy_name == 'nonlinear':
        ntype = extra
        params = strat_params
        def predict(X_test):
            return _predict_nonlinear(X_test, ntype, params)
    
    # Final safety: ensure predictions are finite
    def safe_predict(X_test):
        preds = predict(X_test)
        mask = ~np.isfinite(preds)
        if np.any(mask):
            preds = preds.copy()
            preds[mask] = y_mean
        return preds
    
    return safe_predict
# EVOLVE-BLOCK-END
