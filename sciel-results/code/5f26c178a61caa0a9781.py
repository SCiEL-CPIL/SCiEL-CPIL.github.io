import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280

    X = X_train
    y = y_train
    n = len(y)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Define models
    models = []
    
    # 1. GP Model
    def make_gp_model(l, sigma_f):
        K = cdist(X, X, metric='sqeuclidean')
        K = sigma_f**2 * np.exp(-K / (2 * l**2))
        K[np.diag_indices_from(K)] += 1e-8
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        except:
            return None
        def predict(X_pred):
            if X_pred.ndim == 1:
                X_pred = X_pred.reshape(-1, 1)
            K_pred = cdist(X_pred, X, metric='sqeuclidean')
            K_pred = sigma_f**2 * np.exp(-K_pred / (2 * l**2))
            return K_pred @ alpha
        return predict

    # 2. Polynomial Ridge Model
    def make_poly_model(degree, alpha):
        # Create polynomial features
        X_pow = np.column_stack([X**i for i in range(degree + 1)])
        # Ridge regression
        A = X_pow.T @ X_pow + alpha * np.eye(degree + 1)
        b = X_pow.T @ y
        w = np.linalg.solve(A, b)
        def predict(X_pred):
            if X_pred.ndim == 1:
                X_pred = X_pred.reshape(-1, 1)
            X_pow_pred = np.column_stack([X_pred**i for i in range(degree + 1)])
            return X_pow_pred @ w
        return predict

    # Hyperparameters to try
    # GP: l, sigma_f
    gp_params = []
    for l in [0.5, 1.0, 2.0, 5.0]:
        for sf in [0.5, 1.0, 5.0, 10.0]:
            gp_params.append((l, sf))
            
    # Poly: degree, alpha
    poly_params = []
    for deg in [5, 10, 15, 20]:
        for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            poly_params.append((deg, alpha))

    # Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_model = None
    best_cv_mse = np.inf

    # Evaluate GP candidates
    for l, sf in gp_params:
        if time.time() > deadline:
            break
        cv_errors = []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            # Fit GP on train subset
            K_tr = cdist(X_tr, X_tr, metric='sqeuclidean')
            K_tr = sf**2 * np.exp(-K_tr / (2 * l**2))
            K_tr[np.diag_indices_from(K_tr)] += 1e-8
            try:
                L_tr = np.linalg.cholesky(K_tr)
                alpha_tr = np.linalg.solve(L_tr.T, np.linalg.solve(L_tr, y_tr))
                K_val = cdist(X_val, X_tr, metric='sqeuclidean')
                K_val = sf**2 * np.exp(-K_val / (2 * l**2))
                y_pred = K_val @ alpha_tr
                cv_errors.append(mean_squared_error(y_val, y_pred))
            except:
                cv_errors.append(np.inf)
        
        avg_mse = np.mean(cv_errors)
        if avg_mse < best_cv_mse:
            best_cv_mse = avg_mse
            # Store params to refit later
            best_params = ('gp', l, sf)

    # Evaluate Poly candidates
    for deg, alpha in poly_params:
        if time.time() > deadline:
            break
        cv_errors = []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            X_tr_pow = np.column_stack([X_tr**i for i in range(deg + 1)])
            X_val_pow = np.column_stack([X_val**i for i in range(deg + 1)])
            
            A = X_tr_pow.T @ X_tr_pow + alpha * np.eye(deg + 1)
            b = X_tr_pow.T @ y_tr
            try:
                w = np.linalg.solve(A, b)
                y_pred = X_val_pow @ w
                cv_errors.append(mean_squared_error(y_val, y_pred))
            except:
                cv_errors.append(np.inf)
                
        avg_mse = np.mean(cv_errors)
        if avg_mse < best_cv_mse:
            best_cv_mse = avg_mse
            best_params = ('poly', deg, alpha)

    # Refit best model on full training set
    if best_params is None:
        # Fallback
        w, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), X, X**2]), y, rcond=None)
        return lambda X: (np.column_stack([np.ones(len(X)), X.reshape(-1,1), X.reshape(-1,1)**2]) @ w).flatten()

    if best_params[0] == 'gp':
        l, sf = best_params[1], best_params[2]
        K = cdist(X, X, metric='sqeuclidean')
        K = sf**2 * np.exp(-K / (2 * l**2))
        K[np.diag_indices_from(K)] += 1e-8
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        
        def predict(X_pred):
            if X_pred.ndim == 1:
                X_pred = X_pred.reshape(-1, 1)
            K_pred = cdist(X_pred, X, metric='sqeuclidean')
            K_pred = sf**2 * np.exp(-K_pred / (2 * l**2))
            return K_pred @ alpha
        return predict
    else:
        deg, alpha = best_params[1], best_params[2]
        X_pow = np.column_stack([X**i for i in range(deg + 1)])
        A = X_pow.T @ X_pow + alpha * np.eye(deg + 1)
        b = X_pow.T @ y
        w = np.linalg.solve(A, b)
        
        def predict(X_pred):
            if X_pred.ndim == 1:
                X_pred = X_pred.reshape(-1, 1)
            X_pow_pred = np.column_stack([X_pred**i for i in range(deg + 1)])
            return X_pow_pred @ w
        return predict
