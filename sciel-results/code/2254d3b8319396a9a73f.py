import numpy as np
import time
from scipy.interpolate import CubicSpline

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Since all inputs are 1D, we can exploit this.
    # If d > 1, we fall back to a general method.
    
    def get_rich_features(X_1d):
        x = X_1d.flatten()
        n = len(x)
        cols = [np.ones(n)]
        # Polynomials
        xk = np.ones(n)
        for k in range(1, 20):
            xk = xk * x
            cols.append(xk.copy())
        # Trig
        for freq in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
            cols.append(np.sin(freq * x))
            cols.append(np.cos(freq * x))
        # Exp
        for c in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]:
            cols.append(np.exp(c * x))
        # Other
        cols.append(np.abs(x))
        cols.append(np.sqrt(np.abs(x) + 1e-9))
        cols.append(np.log(np.abs(x) + 1e-9))
        cols.append(1.0 / (x + 1e-9))
        return np.column_stack(cols)

    # Model 1: Cubic Spline (Best for smooth 1D functions)
    def create_spline_model(X_1d, y_1d):
        try:
            # Ensure X is sorted for spline
            idx = np.argsort(X_1d.flatten())
            X_sorted = X_1d[idx]
            y_sorted = y_1d[idx]
            cs = CubicSpline(X_sorted.flatten(), y_sorted)
            def predict(X_new):
                return cs(X_new.flatten()).reshape(X_new.shape[0])
            return predict
        except Exception:
            return None

    # Model 2: Regularized Least Squares (Ridge-like) with Rich Features
    def create_ridge_model(X_1d, y_1d):
        try:
            Xf = get_rich_features(X_1d)
            # Add intercept column if not present (get_rich_features adds ones)
            # Regularization: (X^T X + alpha*I) w = X^T y
            # We set alpha small. Since first col is ones, we might want to not regularize intercept?
            # Simple approach: regularize all, or just use lstsq with small ridge.
            # Let's use np.linalg.lstsq on [Xf, lambda*I] for simplicity or solve directly.
            alpha = 1.0
            n_features = Xf.shape[1]
            # Identity matrix for regularization
            I = np.eye(n_features)
            # Do not regularize the intercept (first column) to allow proper bias
            I[0, 0] = 0.0
            
            A = Xf.T @ Xf + alpha * I
            b = Xf.T @ y_1d
            
            try:
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            def predict(X_new):
                Xf_new = get_rich_features(X_new)
                return Xf_new @ w
            return predict
        except Exception:
            return None

    # Model 3: RBF Interpolation (Gaussian) implemented with NumPy
    def create_rbf_model(X_1d, y_1d):
        try:
            x = X_1d.flatten()
            n = len(x)
            epsilon = 1.0
            
            # Compute RBF matrix
            # K[i,j] = exp(-epsilon^2 * (x_i - x_j)^2)
            diff = x[:, None] - x[None, :]
            K = np.exp(-(epsilon ** 2) * (diff ** 2))
            
            # Solve K @ w = y
            # Add small regularization for stability
            reg = 1e-8 * np.eye(n)
            K_reg = K + reg
            
            try:
                w = np.linalg.solve(K_reg, y_1d)
            except np.linalg.LinAlgError:
                w, _, _, _ = np.linalg.lstsq(K_reg, y_1d, rcond=None)
            
            def predict(X_new):
                x_new = np.asarray(X_new, dtype=np.float64).flatten()
                diff_new = x_new[:, None] - x[None, :]
                K_new = np.exp(-(epsilon ** 2) * (diff_new ** 2))
                return K_new @ w
            return predict
        except Exception:
            return None

    # If d == 1, use the 1D specific models with validation
    if d == 1:
        # Split for validation
        # Use a fixed split for determinism, or random
        idx = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(idx)
        val_size = 20
        val_idx = idx[:val_size]
        train_idx = idx[val_size:]
        
        X_val = X[val_idx]
        y_val = y[val_idx]
        X_tr = X[train_idx]
        y_tr = y[train_idx]
        
        best_model = None
        best_val_mse = np.inf
        
        # Try Spline
        spline_model = create_spline_model(X_tr, y_tr)
        if spline_model:
            try:
                pred_val = spline_model(X_val)
                mse = np.mean((pred_val - y_val)**2)
                if np.isfinite(mse) and mse < best_val_mse:
                    best_val_mse = mse
                    best_model = 'spline'
            except Exception:
                pass
        
        # Try Ridge
        ridge_model = create_ridge_model(X_tr, y_tr)
        if ridge_model:
            try:
                pred_val = ridge_model(X_val)
                mse = np.mean((pred_val - y_val)**2)
                if np.isfinite(mse) and mse < best_val_mse:
                    best_val_mse = mse
                    best_model = 'ridge'
            except Exception:
                pass
            
        # Try RBF
        rbf_model = create_rbf_model(X_tr, y_tr)
        if rbf_model:
            try:
                pred_val = rbf_model(X_val)
                mse = np.mean((pred_val - y_val)**2)
                if np.isfinite(mse) and mse < best_val_mse:
                    best_val_mse = mse
                    best_model = 'rbf'
            except Exception:
                pass

        # Refit best model on all data
        if best_model == 'spline':
            final_model = create_spline_model(X, y)
        elif best_model == 'ridge':
            final_model = create_ridge_model(X, y)
        elif best_model == 'rbf':
            final_model = create_rbf_model(X, y)
        else:
            # Fallback: Linear regression
            try:
                X_aug = np.column_stack([np.ones(n), X])
                w, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
                final_model = lambda X_new: np.column_stack([np.ones(len(X_new)), X_new]) @ w
            except Exception:
                y_mean = np.mean(y)
                final_model = lambda X_new: np.full(len(X_new), y_mean)
        
        def predict(X_new):
            return final_model(np.asarray(X_new, dtype=np.float64))
            
        return predict
    
    else:
        # Fallback for d > 1 (though task says d=1)
        # Since task specifies 1 input, this branch is unlikely to be hit, 
        # but we provide a simple fallback.
        try:
            # Simple Linear Regression fallback
            X_aug = np.column_stack([np.ones(n), X])
            w, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
            def predict(X_new):
                X_new = np.asarray(X_new, dtype=np.float64)
                X_new_aug = np.column_stack([np.ones(X_new.shape[0]), X_new])
                return X_new_aug @ w
            return predict
        except Exception:
            y_mean = np.mean(y)
            def predict(X_new):
                return np.full(len(X_new), y_mean)
            return predict
