import numpy as np
import time
from scipy.linalg import solve

def _make_predict_poly(coeffs, x_min, x_range):
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=float)
        if X_test.ndim == 1:
            x = X_test
        else:
            x = X_test[:, 0]
        x_norm = (x - x_min) / x_range
        # Horner's method for polynomial evaluation
        result = np.full_like(x_norm, coeffs[-1], dtype=float)
        for c in coeffs[-2::-1]:
            result = result * x_norm + c
        return result
    return predict

def _make_predict_rbf(centers, coeffs, gamma, x_min, x_range):
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=float)
        if X_test.ndim == 1:
            x = X_test
        else:
            x = X_test[:, 0]
        x_norm = (x - x_min) / x_range
        diff = x_norm[:, np.newaxis] - centers[np.newaxis, :]
        K_test = np.exp(-gamma * diff**2)
        return K_test @ coeffs
    return predict

def _make_predict_fourier(centers_freq, sin_coeffs, cos_coeffs, x_min, x_range):
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=float)
        if X_test.ndim == 1:
            x = X_test
        else:
            x = X_test[:, 0]
        x_norm = (x - x_min) / x_range
        result = np.full_like(x_norm, sin_coeffs[0], dtype=float) if len(sin_coeffs) > 0 else np.zeros_like(x_norm)
        # Actually, let's store intercept separately or handle in coeffs
        # Let's restructure: coeffs = [a0, b1, a1, b2, a2, ...] or similar
        # Simpler: store all coeffs in one array and features in order
        pass
    return predict

def _fit_rich_poly(X, y, deg, lam=1e-6):
    """Fit a polynomial of degree deg with ridge regularization on normalized x."""
    x = X.flatten()
    x_min = x.min()
    x_max = x.max()
    x_range = x_max - x_min
    if x_range < 1e-10:
        x_range = 1.0
    x_norm = (x - x_min) / x_range
    
    # Build feature matrix
    cols = [np.ones_like(x_norm)]
    for i in range(1, deg + 1):
        cols.append(x_norm ** i)
    A = np.column_stack(cols)
    
    try:
        # Ridge regression
        coeffs = np.linalg.solve(A.T @ A + lam * np.eye(deg + 1), A.T @ y)
        
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=float)
            if X_test.ndim == 1:
                xt = X_test
            else:
                xt = X_test[:, 0]
            xt_norm = (xt - x_min) / x_range
            # Horner
            result = np.full_like(xt_norm, coeffs[-1], dtype=float)
            for c in coeffs[-2::-1]:
                result = result * xt_norm + c
            return result
        
        y_pred = predict(x)
        mse = np.mean((y_pred - y)**2)
        var_y = np.var(y)
        if var_y < 1e-10:
            var_y = 1.0
        score = mse / var_y
        return predict, score
    except Exception:
        return None, np.inf

def _fit_rbf_normalized(X, y, n_centers, gamma, lam=1e-4):
    """Fit RBF with evenly spaced centers in normalized space."""
    x = X.flatten()
    x_min = x.min()
    x_max = x.max()
    x_range = x_max - x_min
    if x_range < 1e-10:
        x_range = 1.0
    x_norm = (x - x_min) / x_range
    
    centers = np.linspace(0, 1, n_centers)
    
    # Kernel matrix
    diff = x_norm[:, np.newaxis] - centers[np.newaxis, :]
    K = np.exp(-gamma * diff**2)
    
    try:
        coeffs = np.linalg.solve(K.T @ K + lam * np.eye(n_centers), K.T @ y)
        
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=float)
            if X_test.ndim == 1:
                xt = X_test
            else:
                xt = X_test[:, 0]
            xt_norm = (xt - x_min) / x_range
            diff = xt_norm[:, np.newaxis] - centers[np.newaxis, :]
            K_test = np.exp(-gamma * diff**2)
            return K_test @ coeffs
        
        y_pred = predict(x)
        mse = np.mean((y_pred - y)**2)
        var_y = np.var(y)
        if var_y < 1e-10:
            var_y = 1.0
        score = mse / var_y
        return predict, score
    except Exception:
        return None, np.inf

def _fit_fourier(X, y, n_terms, lam=1e-6):
    """Fit Fourier series on normalized x in [0, 1]."""
    x = X.flatten()
    x_min = x.min()
    x_max = x.max()
    x_range = x_max - x_min
    if x_range < 1e-10:
        x_range = 1.0
    x_norm = (x - x_min) / x_range
    
    # Features: 1, sin(2*pi*k*x), cos(2*pi*k*x) for k=1..n_terms
    cols = [np.ones_like(x_norm)]
    for k in range(1, n_terms + 1):
        angle = 2 * np.pi * k * x_norm
        cols.append(np.sin(angle))
        cols.append(np.cos(angle))
    
    A = np.column_stack(cols)
    
    try:
        coeffs = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
        
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=float)
            if X_test.ndim == 1:
                xt = X_test
            else:
                xt = X_test[:, 0]
            xt_norm = (xt - x_min) / x_range
            cols = [np.ones_like(xt_norm)]
            for k in range(1, n_terms + 1):
                angle = 2 * np.pi * k * xt_norm
                cols.append(np.sin(angle))
                cols.append(np.cos(angle))
            A_test = np.column_stack(cols)
            return A_test @ coeffs
        
        y_pred = predict(x)
        mse = np.mean((y_pred - y)**2)
        var_y = np.var(y)
        if var_y < 1e-10:
            var_y = 1.0
        score = mse / var_y
        return predict, score
    except Exception:
        return None, np.inf

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    
    # Flatten for 1D
    if X.ndim == 2:
        x = X[:, 0]
    else:
        x = X.flatten()
    y = y.flatten()
    
    best_predict = None
    best_score = np.inf
    
    # Strategy 1: Rich polynomial fits (normalized, ridge)
    for deg in [4, 6, 8, 10, 12, 14, 16]:
        if time.time() > deadline - 30:
            break
        pred, score = _fit_rich_poly(x, y, deg, lam=1e-6)
        if pred is not None and score < best_score:
            best_score = score
            best_predict = pred
    
    # Strategy 2: RBF with evenly spaced centers (normalized)
    for n_centers in [10, 15, 20, 25, 30]:
        for gamma in [5, 10, 20, 50, 100]:
            if time.time() > deadline - 30:
                break
            pred, score = _fit_rbf_normalized(x, y, n_centers, gamma, lam=1e-4)
            if pred is not None and score < best_score:
                best_score = score
                best_predict = pred
    
    # Strategy 3: Fourier series (normalized)
    for n_terms in [5, 10, 15, 20]:
        if time.time() > deadline - 30:
            break
        pred, score = _fit_fourier(x, y, n_terms, lam=1e-6)
        if pred is not None and score < best_score:
            best_score = score
            best_predict = pred
    
    # Fallback: Mean
    if best_predict is None:
        mean_y = np.mean(y)
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=float)
            if X_test.ndim == 1:
                return np.full(X_test.shape, mean_y)
            else:
                return np.full(X_test.shape[0], mean_y)
        best_predict = predict
    
    # Wrap to handle input shape
    def predict(X_test):
        return best_predict(X_test)
    
    return predict
