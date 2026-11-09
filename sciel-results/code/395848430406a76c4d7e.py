import numpy as np
import time
from scipy.interpolate import CubicSpline, PchipInterpolator

def fit(X_train, y_train):
    t0 = time.time()
    deadline = t0 + 280
    
    X = X_train
    y = y_train
    n, d = X.shape
    x = X[:, 0]
    
    # Sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    var_y = np.var(y)
    if var_y < 1e-15:
        return lambda X: np.zeros(len(X))
    
    # Strategy 1: Cubic spline interpolation (s=0 equivalent)
    # For noiseless smooth data, this should be near-perfect
    try:
        cs = CubicSpline(x_sorted, y_sorted, bc_type='natural')
        spline_pred = cs(x_sorted)  # sanity check
        spline_valid = np.all(np.isfinite(spline_pred))
    except Exception:
        cs = None
        spline_valid = False
    
    # Strategy 2: PCHIP (monotone-preserving, no overshoot)
    try:
        pchip = PchipInterpolator(x_sorted, y_sorted)
        pchip_pred = pchip(x_sorted)
        pchip_valid = np.all(np.isfinite(pchip_pred))
    except Exception:
        pchip = None
        pchip_valid = False
    
    # Strategy 3: Rich basis + ridge regression
    def make_features(x):
        p1 = np.ones_like(x)
        p2 = x
        p3 = x**2
        p4 = x**3
        p5 = x**4
        p6 = x**5
        p7 = x**6
        t1 = np.sin(x)
        t2 = np.cos(x)
        t3 = np.sin(2*x)
        t4 = np.cos(2*x)
        t5 = np.sin(3*x)
        t6 = np.cos(3*x)
        t7 = np.sin(4*x)
        t8 = np.cos(4*x)
        t9 = np.sin(5*x)
        t10 = np.cos(5*x)
        t11 = np.sin(6*x)
        t12 = np.cos(6*x)
        e1 = np.exp(x)
        e2 = np.exp(-x)
        e3 = np.exp(2*x)
        e4 = np.exp(-2*x)
        r1 = np.sqrt(np.abs(x) + 1e-10)
        r2 = np.log(np.abs(x) + 1e-10)
        r3 = 1.0 / (x + 1e-10)
        r4 = 1.0 / (x**2 + 1e-10)
        return np.column_stack([p1, p2, p3, p4, p5, p6, p7,
                                  t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12,
                                  e1, e2, e3, e4,
                                  r1, r2, r3, r4])
    
    ridge_w = None
    ridge_valid = False
    try:
        F_train = make_features(x)
        # Use ridge with small alpha to handle collinearity
        alpha = 1e-6
        A = F_train.T @ F_train + alpha * np.eye(F_train.shape[1])
        b = F_train.T @ y
        ridge_w = np.linalg.solve(A, b)
        ridge_pred = F_train @ ridge_w
        ridge_valid = np.all(np.isfinite(ridge_pred))
    except Exception:
        ridge_w = None
        ridge_valid = False
    
    # Evaluate each strategy on training data (since data is noiseless,
    # a good model should fit training data very well)
    # Use a held-out subset for more honest evaluation
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_val = max(20, n // 5)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    
    x_val = x[val_idx]
    y_val = y[val_idx]
    x_tr = x[tr_idx]
    y_tr = y[tr_idx]
    
    # Evaluate spline on held-out
    spline_cv_err = np.inf
    if cs is not None:
        try:
            # Refit on training only for CV
            cs_tr = CubicSpline(x_tr, y_tr, bc_type='natural')
            pred = cs_tr(x_val)
            if np.all(np.isfinite(pred)):
                spline_cv_err = np.mean((pred - y_val)**2)
        except Exception:
            pass
    
    # Evaluate pchip on held-out
    pchip_cv_err = np.inf
    if pchip is not None:
        try:
            pchip_tr = PchipInterpolator(x_tr, y_tr)
            pred = pchip_tr(x_val)
            if np.all(np.isfinite(pred)):
                pchip_cv_err = np.mean((pred - y_val)**2)
        except Exception:
            pass
    
    # Evaluate ridge on held-out
    ridge_cv_err = np.inf
    if ridge_w is not None:
        try:
            F_tr = make_features(x_tr)
            alpha = 1e-6
            A = F_tr.T @ F_tr + alpha * np.eye(F_tr.shape[1])
            b = F_tr.T @ y_tr
            w_tr = np.linalg.solve(A, b)
            F_val = make_features(x_val)
            pred = F_val @ w_tr
            if np.all(np.isfinite(pred)):
                ridge_cv_err = np.mean((pred - y_val)**2)
        except Exception:
            pass
    
    # Pick best strategy
    strategies = []
    if cs is not None and spline_valid:
        strategies.append(('spline', spline_cv_err, cs))
    if pchip is not None and pchip_valid:
        strategies.append(('pchip', pchip_cv_err, pchip))
    if ridge_w is not None and ridge_valid:
        strategies.append(('ridge', ridge_cv_err, ridge_w))
    
    if not strategies:
        # Absolute fallback: constant prediction
        mean_y = np.mean(y)
        return lambda X: np.full(len(X), mean_y)
    
    strategies.sort(key=lambda s: s[1])
    best_name, best_err, best_obj = strategies[0]
    
    # If the best error is already very small, great
    # If all errors are large, still pick the best
    
    if best_name == 'spline':
        def predict(X):
            xq = X[:, 0]
            return best_obj(xq)
        return predict
    elif best_name == 'pchip':
        def predict(X):
            xq = X[:, 0]
            return best_obj(xq)
        return predict
    else:  # ridge
        def predict(X):
            xq = X[:, 0]
            F = make_features(xq)
            return F @ best_obj
        return predict
