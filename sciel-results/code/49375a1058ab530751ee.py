import numpy as np
import time
from scipy.interpolate import UnivariateSpline

_START_TIME = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def fit(X_train, y_train):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    X = X_train
    y = y_train
    n, d = X.shape
    
    # Standardize X
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma[sigma == 0] = 1.0
    X_scaled = (X - mu) / sigma
    
    best_cv_error = np.inf
    best_model = None
    best_params = None
    
    n_folds = 5
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    def cv_error(model_type, params):
        errors = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            X_tr = X_scaled[train_idx]
            y_tr = y[train_idx]
            X_val = X_scaled[val_idx]
            y_val = y[val_idx]
            
            if model_type == 'poly':
                deg, lam = params
                X_tr_f = np.column_stack([X_tr[:, 0]**k for k in range(deg+1)])
                X_val_f = np.column_stack([X_val[:, 0]**k for k in range(deg+1)])
                A = X_tr_f.T @ X_tr_f + lam * np.eye(deg+1)
                b = X_tr_f.T @ y_tr
                try:
                    w = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    w = np.linalg.lstsq(X_tr_f, y_tr, rcond=None)[0]
                pred = X_val_f @ w
                
            elif model_type == 'trig':
                K, lam = params
                x_tr = X_tr[:, 0]
                x_val = X_val[:, 0]
                feats_tr = [np.ones(len(x_tr))]
                feats_val = [np.ones(len(x_val))]
                for k in range(1, K+1):
                    feats_tr.append(np.sin(k*x_tr))
                    feats_tr.append(np.cos(k*x_tr))
                    feats_val.append(np.sin(k*x_val))
                    feats_val.append(np.cos(k*x_val))
                X_tr_f = np.column_stack(feats_tr)
                X_val_f = np.column_stack(feats_val)
                n_feats = X_tr_f.shape[1]
                A = X_tr_f.T @ X_tr_f + lam * np.eye(n_feats)
                b = X_tr_f.T @ y_tr
                try:
                    w = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    w = np.linalg.lstsq(X_tr_f, y_tr, rcond=None)[0]
                pred = X_val_f @ w
                
            elif model_type == 'rbf':
                gamma, lam = params
                def rbf_kernel(X1, X2, gamma):
                    sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
                    sq_dists = np.maximum(sq_dists, 0)
                    return np.exp(-gamma * sq_dists)
                K_tr = rbf_kernel(X_tr, X_tr, gamma)
                K_val = rbf_kernel(X_val, X_tr, gamma)
                A = K_tr + lam * np.eye(len(X_tr))
                b = y_tr
                try:
                    alpha = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    alpha = np.linalg.lstsq(K_tr, y_tr, rcond=None)[0]
                pred = K_val @ alpha
                
            elif model_type == 'spline':
                s_val = params
                sort_idx = np.argsort(X_tr[:, 0])
                X_tr_sorted = X_tr[sort_idx, 0]
                y_tr_sorted = y_tr[sort_idx]
                try:
                    sp = UnivariateSpline(X_tr_sorted, y_tr_sorted, s=s_val, k=3)
                    pred = sp(X_val[:, 0])
                except Exception:
                    return np.inf
            else:
                return np.inf
            
            err = np.mean((pred - y_val)**2)
            errors.append(err)
            
        return np.mean(errors)

    # Search Poly
    for deg in range(1, 11):
        if _time_left() < 5.0: break
        for lam in np.logspace(-5, 2, 10):
            err = cv_error('poly', (deg, lam))
            if err < best_cv_error:
                best_cv_error = err
                best_model = 'poly'
                best_params = (deg, lam)

    # Search Trig
    for K in range(1, 11):
        if _time_left() < 5.0: break
        for lam in np.logspace(-5, 2, 10):
            err = cv_error('trig', (K, lam))
            if err < best_cv_error:
                best_cv_error = err
                best_model = 'trig'
                best_params = (K, lam)

    # Search RBF
    for gamma in np.logspace(-3, 2, 15):
        if _time_left() < 5.0: break
        for lam in np.logspace(-5, 2, 10):
            err = cv_error('rbf', (gamma, lam))
            if err < best_cv_error:
                best_cv_error = err
                best_model = 'rbf'
                best_params = (gamma, lam)
                
    # Search Spline
    for s_val in np.logspace(-5, 5, 20):
        if _time_left() < 5.0: break
        err = cv_error('spline', s_val)
        if err < best_cv_error:
            best_cv_error = err
            best_model = 'spline'
            best_params = s_val

    # Refit best model on full data
    if best_model == 'poly':
        deg, lam = best_params
        X_f = np.column_stack([X_scaled[:, 0]**k for k in range(deg+1)])
        A = X_f.T @ X_f + lam * np.eye(deg+1)
        b = X_f.T @ y
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(X_f, y, rcond=None)[0]
            
        def predict(X):
            Xs = (X - mu) / sigma
            Xf = np.column_stack([Xs[:, 0]**k for k in range(deg+1)])
            return Xf @ w
            
    elif best_model == 'trig':
        K, lam = best_params
        x = X_scaled[:, 0]
        feats = [np.ones(len(x))]
        for k in range(1, K+1):
            feats.append(np.sin(k*x))
            feats.append(np.cos(k*x))
        X_f = np.column_stack(feats)
        n_feats = X_f.shape[1]
        A = X_f.T @ X_f + lam * np.eye(n_feats)
        b = X_f.T @ y
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(X_f, y, rcond=None)[0]
            
        def predict(X):
            Xs = (X - mu) / sigma
            xv = Xs[:, 0]
            feats = [np.ones(len(xv))]
            for k in range(1, K+1):
                feats.append(np.sin(k*xv))
                feats.append(np.cos(k*xv))
            Xf = np.column_stack(feats)
            return Xf @ w
            
    elif best_model == 'rbf':
        gamma, lam = best_params
        def rbf_kernel(X1, X2, gamma):
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            return np.exp(-gamma * sq_dists)
            
        X_tr_scaled = X_scaled
        K_tr = rbf_kernel(X_tr_scaled, X_tr_scaled, gamma)
        A = K_tr + lam * np.eye(n)
        b = y
        try:
            alpha = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(K_tr, y, rcond=None)[0]
            
        X_train_scaled_store = X_tr_scaled.copy()
        alpha_store = alpha.copy()
        
        def predict(X):
            Xs = (X - mu) / sigma
            K_val = rbf_kernel(Xs, X_train_scaled_store, gamma)
            return K_val @ alpha_store

    elif best_model == 'spline':
        s_val = best_params
        sort_idx = np.argsort(X_scaled[:, 0])
        X_tr_sorted = X_scaled[sort_idx, 0]
        y_tr_sorted = y[sort_idx]
        try:
            sp = UnivariateSpline(X_tr_sorted, y_tr_sorted, s=s_val, k=3)
        except Exception:
            # Fallback to linear if spline fails
            def predict(X):
                return np.full(len(X), np.mean(y))
            return predict
            
        def predict(X):
            Xs = (X - mu) / sigma
            return sp(Xs[:, 0])

    else:
        def predict(X):
            return np.full(len(X), np.mean(y))

    return predict
