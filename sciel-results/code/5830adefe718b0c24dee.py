"""Symbolic regression via adaptive model selection with spline interpolation for 1D noiseless data."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize_scalar

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 270  # Safety margin for 4 equations (300s total)
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n = len(y)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    d = X.shape[1]
    
    def time_left():
        return deadline - time.time()
    
    # --- Strategy 1: Cubic spline interpolation (best for smooth 1D noiseless data) ---
    # Sort by x
    if d == 1:
        x = X[:, 0]
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        
        # Check if data is roughly monotonic or smooth (low second derivative)
        # Try cubic spline and evaluate on held-out points
        rng = np.random.RandomState(42)
        indices = np.arange(n)
        rng.shuffle(indices)
        split = int(0.8 * n)
        idx_train = indices[:split]
        idx_val = indices[split:]
        
        x_tr = x[idx_train]
        y_tr = y[idx_train]
        x_val = x[idx_val]
        y_val = y[idx_val]
        
        sort_tr = np.argsort(x_tr)
        x_tr_s = x_tr[sort_tr]
        y_tr_s = y_tr[sort_tr]
        
        try:
            cs = CubicSpline(x_tr_s, y_tr_s, bc_type='natural')
            y_pred_val = cs(x_val)
            spline_val_mse = np.mean((y_pred_val - y_val)**2)
        except:
            spline_val_mse = np.inf
        
        # --- Strategy 2: Global RBF with training points as centers ---
        # Use all training points as RBF centers
        x_all_sorted = x[sort_idx]
        y_all_sorted = y[sort_idx]
        
        def rbf_kernel_1d(x1, x2, width):
            """Gaussian RBF kernel in 1D."""
            diff = x1[:, None] - x2[None, :]
            return np.exp(-0.5 * (diff / width)**2)
        
        best_rbf_val_mse = np.inf
        best_rbf_width = 1.0
        best_rbf_weights = None
        
        # Try different widths
        x_range = x.max() - x.min()
        if x_range < 1e-10:
            x_range = 1.0
        
        widths = [x_range * f for f in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]]
        widths = sorted(set(widths), key=lambda w: w)
        
        for width in widths:
            if time_left() < 5:
                break
            try:
                # Build RBF matrix for training
                K_tr = rbf_kernel_1d(x_tr, x_tr_s, width)
                # Add small regularization for numerical stability
                lam = 1e-6
                A = K_tr + lam * np.eye(len(x_tr))
                w = np.linalg.solve(A, y_tr)
                
                # Evaluate on validation
                K_val = rbf_kernel_1d(x_val, x_tr_s, width)
                y_pred_val = K_val @ w
                mse = np.mean((y_pred_val - y_val)**2)
                
                if mse < best_rbf_val_mse:
                    best_rbf_val_mse = mse
                    best_rbf_width = width
                    best_rbf_weights = w
            except np.linalg.LinAlgError:
                continue
        
        # --- Strategy 3: Sparse RBF (subset of centers) ---
        # Use fewer centers for better conditioning
        best_sparse_val_mse = np.inf
        best_sparse_width = 1.0
        best_sparse_centers = None
        best_sparse_weights = None
        
        # Try different numbers of centers
        n_centers_options = [20, 50, 100, 150]
        for n_c in n_centers_options:
            if time_left() < 5:
                break
            if n_c >= len(x_tr_s):
                continue
            # Uniformly sample centers from training data
            center_idx = np.linspace(0, len(x_tr_s) - 1, n_c, dtype=int)
            centers = x_tr_s[center_idx]
            
            for width in [x_range * f for f in [0.2, 0.3, 0.5, 0.7, 1.0]]:
                try:
                    K_tr_sp = rbf_kernel_1d(x_tr, centers, width)
                    lam = 1e-5
                    A_sp = K_tr_sp.T @ K_tr_sp + lam * np.eye(n_c)
                    b_sp = K_tr_sp.T @ y_tr
                    w_sp = np.linalg.solve(A_sp, b_sp)
                    
                    K_val_sp = rbf_kernel_1d(x_val, centers, width)
                    y_pred_val = K_val_sp @ w_sp
                    mse = np.mean((y_pred_val - y_val)**2)
                    
                    if mse < best_sparse_val_mse:
                        best_sparse_val_mse = mse
                        best_sparse_width = width
                        best_sparse_centers = centers
                        best_sparse_weights = w_sp
                except np.linalg.LinAlgError:
                    continue
        
        # --- Select best model ---
        # Compare all strategies
        candidates = [
            ('spline', spline_val_mse),
            ('rbf', best_rbf_val_mse),
            ('sparse_rbf', best_sparse_val_mse)
        ]
        
        candidates.sort(key=lambda x: x[1])
        best_model = candidates[0][0]
        best_val_mse = candidates[0][1]
        
        # Refit on all data with best model
        if best_model == 'spline':
            try:
                cs_final = CubicSpline(x_sorted, y_sorted, bc_type='natural')
                def predict(X_pred):
                    X_pred = np.asarray(X_pred, dtype=np.float64)
                    if X_pred.ndim == 1:
                        X_pred = X_pred.reshape(-1, 1)
                    return cs_final(X_pred[:, 0])
            except:
                # Fallback to nearest neighbor
                def predict(X_pred):
                    X_pred = np.asarray(X_pred, dtype=np.float64)
                    if X_pred.ndim == 1:
                        X_pred = X_pred.reshape(-1, 1)
                    x_pred = X_pred[:, 0]
                    # Find nearest training point
                    dists = np.abs(x_pred[:, None] - x_sorted[None, :])
                    nearest_idx = np.argmin(dists, axis=1)
                    return y_sorted[nearest_idx]
        
        elif best_model == 'rbf':
            width = best_rbf_width
            # Refit on all data
            K_all = rbf_kernel_1d(x, x_sorted, width)
            lam = 1e-6
            A_all = K_all + lam * np.eye(len(x))
            try:
                w_final = np.linalg.solve(A_all, y)
                def predict(X_pred):
                    X_pred = np.asarray(X_pred, dtype=np.float64)
                    if X_pred.ndim == 1:
                        X_pred = X_pred.reshape(-1, 1)
                    x_pred = X_pred[:, 0]
                    K_pred = rbf_kernel_1d(x_pred, x_sorted, width)
                    return K_pred @ w_final
            except:
                # Fallback
                def predict(X_pred):
                    X_pred = np.asarray(X_pred, dtype=np.float64)
                    if X_pred.ndim == 1:
                        X_pred = X_pred.reshape(-1, 1)
                    x_pred = X_pred[:, 0]
                    dists = np.abs(x_pred[:, None] - x_sorted[None, :])
                    nearest_idx = np.argmin(dists, axis=1)
                    return y_sorted[nearest_idx]
        
        elif best_model == 'sparse_rbf':
            width = best_sparse_width
            centers = best_sparse_centers
            # Refit on all data
            K_all_sp = rbf_kernel_1d(x, centers, width)
            lam = 1e-5
            A_sp_all = K_all_sp.T @ K_all_sp + lam * np.eye(len(centers))
            b_sp_all = K_all_sp.T @ y
            try:
                w_sp_final = np.linalg.solve(A_sp_all, b_sp_all)
                def predict(X_pred):
                    X_pred = np.asarray(X_pred, dtype=np.float64)
                    if X_pred.ndim == 1:
                        X_pred = X_pred.reshape(-1, 1)
                    x_pred = X_pred[:, 0]
                    K_pred_sp = rbf_kernel_1d(x_pred, centers, width)
                    return K_pred_sp @ w_sp_final
            except:
                def predict(X_pred):
                    X_pred = np.asarray(X_pred, dtype=np.float64)
                    if X_pred.ndim == 1:
                        X_pred = X_pred.reshape(-1, 1)
                    x_pred = X_pred[:, 0]
                    dists = np.abs(x_pred[:, None] - x_sorted[None, :])
                    nearest_idx = np.argmin(dists, axis=1)
                    return y_sorted[nearest_idx]
        
        else:
            # Ultimate fallback
            def predict(X_pred):
                X_pred = np.asarray(X_pred, dtype=np.float64)
                if X_pred.ndim == 1:
                    X_pred = X_pred.reshape(-1, 1)
                x_pred = X_pred[:, 0]
                dists = np.abs(x_pred[:, None] - x_sorted[None, :])
                nearest_idx = np.argmin(dists, axis=1)
                return y_sorted[nearest_idx]
        
        return predict
    
    else:
        # Multi-dimensional fallback: KRR with RBF
        def rbf_kernel_md(X1, X2, gamma):
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
            np.maximum(sq_dists, 0, out=sq_dists)
            return np.exp(-gamma * sq_dists)
        
        rng = np.random.RandomState(42)
        indices = np.arange(n)
        rng.shuffle(indices)
        split = int(0.8 * n)
        idx_train = indices[:split]
        idx_val = indices[split:]
        
        X_tr = X[idx_train]
        y_tr = y[idx_train]
        X_val = X[idx_val]
        y_val = y[idx_val]
        
        best_val_mse = np.inf
        best_gamma = 1.0
        best_alpha = None
        
        gammas = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
        for gamma in gammas:
            if time_left() < 5:
                break
            try:
                K_tr = rbf_kernel_md(X_tr, X_tr, gamma)
                lam = 1e-4
                A = K_tr + lam * np.eye(len(X_tr))
                alpha_vec = np.linalg.solve(A, y_tr)
                
                K_val = rbf_kernel_md(X_val, X_tr, gamma)
                y_pred_val = K_val @ alpha_vec
                mse = np.mean((y_pred_val - y_val)**2)
                
                if mse < best_val_mse:
                    best_val_mse = mse
                    best_gamma = gamma
                    best_alpha = alpha_vec
            except:
                continue
        
        # Refit on all data
        gamma = best_gamma
        K_all = rbf_kernel_md(X, X, gamma)
        lam = 1e-4
        A_all = K_all + lam * np.eye(n)
        try:
            alpha_final = np.linalg.solve(A_all, y)
            def predict(X_pred):
                X_pred = np.asarray(X_pred, dtype=np.float64)
                if X_pred.ndim == 1:
                    X_pred = X_pred.reshape(-1, 1)
                K_pred = rbf_kernel_md(X_pred, X, gamma)
                return K_pred @ alpha_final
        except:
            def predict(X_pred):
                X_pred = np.asarray(X_pred, dtype=np.float64)
                if X_pred.ndim == 1:
                    X_pred = X_pred.reshape(-1, 1)
                return np.full(len(X_pred), np.mean(y))
        
        return predict

# EVOLVE-BLOCK-END
