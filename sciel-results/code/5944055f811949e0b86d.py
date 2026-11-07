import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import cho_factor, cho_solve
import time
import warnings

warnings.filterwarnings('ignore')

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 320.0  # Safety margin before 330s limit
    
    X = np.asarray(X_train).flatten().astype(float)
    y = np.asarray(y_train).flatten().astype(float)
    n = len(X)
    
    # Handle constant case
    if np.var(y) < 1e-12:
        mean_y = np.mean(y)
        def predict_const(X):
            return np.full(np.asarray(X).shape, mean_y)
        return predict_const
    
    # Time check helper
    def time_ok():
        return (time.time() - start_time) < deadline - 5
    
    # Build a rich feature library
    def build_features(x):
        features = [np.ones_like(x)]
        
        # Polynomials up to degree 12
        for d in range(1, 13):
            features.append(x**d)
        
        # Trig functions
        for d in range(1, 8):
            features.append(np.sin(x * d))
            features.append(np.cos(x * d))
            features.append(x * np.sin(x * d))
            features.append(x * np.cos(x * d))
        
        # More trig combinations
        features.append(np.sin(x**2))
        features.append(np.cos(x**2))
        features.append(np.sin(2*x))
        features.append(np.cos(2*x))
        features.append(np.sin(3*x))
        features.append(np.cos(3*x))
        features.append(np.sin(x/2))
        features.append(np.cos(x/2))
        features.append(x**2 * np.sin(x))
        features.append(x**2 * np.cos(x))
        features.append(x**3 * np.sin(x))
        features.append(x**3 * np.cos(x))
        
        # Exp functions
        features.append(np.exp(x))
        features.append(np.exp(-x))
        features.append(np.exp(x**2 / 10))
        features.append(np.exp(-x**2 / 10))
        features.append(np.exp(x**2 / 20))
        features.append(np.exp(-x**2 / 20))
        features.append(x * np.exp(x))
        features.append(x * np.exp(-x))
        features.append(x**2 * np.exp(-x**2 / 10))
        features.append(x**2 * np.exp(-x**2 / 20))
        features.append(x * np.exp(-x**2))
        features.append(np.exp(-x**2))
        features.append(np.exp(-x**2 / 2))
        features.append(np.exp(-x**2 / 4))
        
        # Log functions
        features.append(np.log1p(np.abs(x)))
        features.append(np.log1p(x**2))
        features.append(np.log1p(x**2 / 2))
        features.append(x * np.log1p(x**2))
        features.append(np.log1p(np.abs(x**2)))
        
        # Rational functions
        features.append(x / (1 + x**2))
        features.append(x / (1 + x**2 / 2))
        features.append(x**2 / (1 + x**2))
        features.append(x**3 / (1 + x**2))
        features.append(x / (1 + x**4))
        features.append(x**2 / (1 + x**4))
        features.append(1 / (1 + x**2))
        features.append(1 / (1 + x**2 / 2))
        
        # Tanh and sigmoid
        features.append(np.tanh(x))
        features.append(np.tanh(2*x))
        features.append(np.tanh(x/2))
        features.append(x * np.tanh(x))
        features.append(x**2 * np.tanh(x))
        features.append(1 / (1 + np.exp(-x)))
        features.append(1 / (1 + np.exp(-2*x)))
        features.append(x / (1 + np.exp(-x)))
        
        # Sqrt functions
        features.append(np.sqrt(1 + x**2))
        features.append(np.sqrt(1 + x**2 / 2))
        features.append(x / np.sqrt(1 + x**2))
        features.append(x**2 / np.sqrt(1 + x**2))
        
        # Absolute value
        features.append(np.abs(x))
        features.append(x * np.abs(x))
        features.append(np.abs(x**2 - 1))
        
        # More combinations
        features.append(np.sin(x) * np.exp(-x**2))
        features.append(np.cos(x) * np.exp(-x**2))
        features.append(np.sin(x) * np.exp(-x**2 / 2))
        features.append(np.cos(x) * np.exp(-x**2 / 2))
        features.append(x * np.sin(x) * np.exp(-x**2))
        features.append(x * np.cos(x) * np.exp(-x**2))
        
        # sin/cos of multiples
        features.append(np.sin(4*x))
        features.append(np.cos(4*x))
        features.append(np.sin(5*x))
        features.append(np.cos(5*x))
        features.append(np.sin(6*x))
        features.append(np.cos(6*x))
        features.append(x * np.sin(2*x))
        features.append(x * np.cos(2*x))
        
        # Higher order
        features.append(x**13)
        features.append(x**14)
        features.append(x**15)
        features.append(np.sin(x**3))
        features.append(np.cos(x**3))
        features.append(np.sin(x**4))
        features.append(np.cos(x**4))
        features.append(x * np.sin(x**2))
        features.append(x * np.cos(x**2))
        
        return np.column_stack(features)
    
    # Build feature matrix
    try:
        F = build_features(X)
        # Remove any non-finite columns
        for j in range(F.shape[1]):
            if not np.all(np.isfinite(F[:, j])):
                F = np.delete(F, j, axis=1)
    except Exception:
        # Fallback to polynomial
        F = np.column_stack([X**d for d in range(12)])
    
    # Strategy 1: Try simple polynomial fits first
    best_mse = np.inf
    best_predictor = None
    
    # Try polynomial fits of various degrees
    for deg in range(1, 11):
        if not time_ok():
            break
        try:
            coef = np.polyfit(X, y, deg)
            y_pred = np.polyval(coef, X)
            mse = np.mean((y_pred - y)**2)
            if mse < best_mse:
                best_mse = mse
                best_predictor = lambda x, c=coef: np.polyval(c, np.asarray(x).flatten()).reshape(np.asarray(x).shape)
            if mse < 1e-12:
                break
        except Exception:
            continue
    
    # Strategy 2: Forward selection with regularization
    if time_ok():
        # Use ridge regression with forward selection
        n_features = F.shape[1]
        
        # Try various lambda values
        for lam in [0.01, 0.1, 1.0, 10.0, 100.0]:
            if not time_ok():
                break
            try:
                # Ridge regression
                A = F.T @ F + lam * np.eye(n_features)
                b = F.T @ y
                w = np.linalg.solve(A, b)
                y_pred = F @ w
                mse = np.mean((y_pred - y)**2)
                
                if mse < best_mse:
                    best_mse = mse
                    best_predictor = lambda x, w=w, F_train=F, X_train=X: None  # placeholder
                    # Need to store the actual predictor
                    F_train_copy = F
                    w_copy = w
                    X_train_copy = X
                    def make_predictor(F_t, w_t, X_t):
                        def pred(X):
                            x = np.asarray(X).flatten().astype(float)
                            F_new = build_features(x)
                            # Handle non-finite
                            F_new = np.nan_to_num(F_new)
                            return (F_new @ w_t).reshape(np.asarray(X).shape)
                        return pred
                    best_predictor = make_predictor(F_train_copy, w_copy, X_train_copy)
            except Exception:
                continue
        
        # Forward selection with subset selection
        if time_ok():
            # Try selecting top-k features by correlation
            y_mean = np.mean(y)
            y_centered = y - y_mean
            F_mean = np.mean(F, axis=0)
            F_centered = F - F_mean
            
            # Compute correlations
            F_norm = np.sqrt(np.sum(F_centered**2, axis=0))
            y_norm = np.sqrt(np.sum(y_centered**2))
            F_norm[F_norm < 1e-10] = 1.0
            y_norm = max(y_norm, 1e-10)
            
            corr = np.sum(F_centered * y_centered, axis=0) / (F_norm * y_norm)
            corr[np.abs(corr) < 1e-10] = 0
            
            # Sort by absolute correlation
            order = np.argsort(-np.abs(corr))
            
            # Try top-k subsets
            for k in [5, 10, 20, 30, 50, 80, 120, min(150, n_features)]:
                if not time_ok():
                    break
                try:
                    top_k = order[:k]
                    F_sub = F[:, top_k]
                    
                    # Ridge regression on subset
                    for lam in [0.1, 1.0, 10.0]:
                        A = F_sub.T @ F_sub + lam * np.eye(k)
                        b = F_sub.T @ y
                        w_sub = np.linalg.solve(A, b)
                        y_pred = F_sub @ w_sub
                        mse = np.mean((y_pred - y)**2)
                        
                        if mse < best_mse:
                            best_mse = mse
                            F_sub_copy = F_sub
                            w_sub_copy = w_sub
                            top_k_copy = top_k
                            def make_predictor_subset(F_s, w_s, tk):
                                def pred(X):
                                    x = np.asarray(X).flatten().astype(float)
                                    F_new = build_features(x)
                                    F_new = np.nan_to_num(F_new)
                                    F_new_sub = F_new[:, tk]
                                    return (F_new_sub @ w_s).reshape(np.asarray(X).shape)
                                return pred
                            best_predictor = make_predictor_subset(F_sub_copy, w_sub_copy, top_k_copy)
                except Exception:
                    continue
    
    # Strategy 3: Try GP with simpler kernel
    if time_ok() and best_mse > 1e-8 * np.var(y):
        # Normalize
        X_mean = np.mean(X)
        X_std = np.std(X)
        if X_std < 1e-10:
            X_std = 1.0
        X_norm = (X - X_mean) / X_std
        
        y_mean = np.mean(y)
        y_std = np.std(y)
        if y_std < 1e-10:
            y_std = 1.0
        y_norm = (y - y_mean) / y_std
        
        def gp_predict(params, X_test):
            ls, var, noise = params
            ls = max(ls, 1e-3)
            var = max(var, 1e-3)
            noise = max(noise, 1e-6)
            
            # Train kernel
            d2 = np.sum(X_norm**2, axis=1)[:, None] + np.sum(X_norm**2, axis=1)[None, :] - 2 * X_norm @ X_norm.T
            d2 = np.maximum(d2, 0)
            K = var * np.exp(-0.5 * d2 / (ls**2)) + noise * np.eye(n)
            
            try:
                L = cho_factor(K + 1e-10 * np.eye(n))
                alpha = cho_solve(L, y_norm)
                
                # Test kernel
                x_test = np.asarray(X_test).flatten().astype(float)
                x_norm = (x_test - X_mean) / X_std
                d2_star = np.sum(x_norm**2, axis=1)[:, None] + np.sum(X_norm**2, axis=1)[None, :] - 2 * x_norm @ X_norm.T
                d2_star = np.maximum(d2_star, 0)
                K_star = var * np.exp(-0.5 * d2_star / (ls**2))
                
                mu = y_mean + y_std * (K_star @ alpha)
                return mu.reshape(np.asarray(X_test).shape)
            except Exception:
                return None
        
        # Try a few GP parameter sets
        for ls, var, noise in [(0.5, 1.0, 0.01), (1.0, 1.0, 0.01), (2.0, 1.0, 0.01), 
                                (0.5, 2.0, 0.01), (1.0, 2.0, 0.01), (2.0, 2.0, 0.01),
                                (0.3, 1.0, 0.001), (1.5, 1.0, 0.001)]:
            if not time_ok():
                break
            try:
                pred = gp_predict([ls, var, noise], X)
                if pred is not None:
                    mse = np.mean((pred - y)**2)
                    if mse < best_mse:
                        best_mse = mse
                        ls_copy, var_copy, noise_copy = ls, var, noise
                        def make_gp_predictor(ls_c, var_c, noise_c):
                            def pred(X):
                                return gp_predict([ls_c, var_c, noise_c], X)
                            return pred
                        best_predictor = make_gp_predictor(ls_copy, var_copy, noise_copy)
            except Exception:
                continue
    
    # If we have a good predictor, use it
    if best_predictor is not None:
        # Final validation
        try:
            y_check = best_predictor(X)
            if np.all(np.isfinite(y_check)):
                return best_predictor
        except Exception:
            pass
    
    # Fallback: high-degree polynomial
    try:
        deg = min(15, n - 1)
        coef = np.polyfit(X, y, deg)
        def predict_fallback(X):
            x = np.asarray(X).flatten().astype(float)
            return np.polyval(coef, x).reshape(np.asarray(X).shape)
        return predict_fallback
    except Exception:
        mean_y = np.mean(y)
        def predict_mean(X):
            return np.full(np.asarray(X).shape, mean_y)
        return predict_mean
