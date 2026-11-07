"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE_MARGIN = 10.0  # seconds before the 330s wall-clock limit to stop
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    def time_left():
        return (start_time + 330.0 - DEADLINE_MARGIN) - time.time()
    
    # Handle constant y
    y_mean = np.mean(y)
    y_var = np.var(y)
    if y_var < 1e-12:
        c = y_mean
        return lambda X: np.full(len(X), c)
    
    # Define a rich set of basis functions
    def get_basis_functions(X):
        n = len(X)
        cols = [np.ones(n)]
        
        # Polynomial up to degree 12
        for p in range(1, 13):
            cols.append(X ** p)
        
        # Trigonometric
        # Try frequencies 1, 2, 3, 4, 5, 6, 7, 8
        for k in range(1, 9):
            cols.append(np.sin(k * X))
            cols.append(np.cos(k * X))
            
        # x * trig
        for k in range(1, 5):
            cols.append(X * np.sin(k * X))
            cols.append(X * np.cos(k * X))
            
        # Exponential
        cols.append(np.exp(X))
        cols.append(np.exp(-X))
        cols.append(np.exp(0.5 * X))
        cols.append(np.exp(-0.5 * X))
        cols.append(np.exp(2 * X))
        cols.append(np.exp(-2 * X))
        cols.append(np.exp(0.1 * X))
        cols.append(np.exp(-0.1 * X))
        
        # Logarithmic
        cols.append(np.log1p(np.abs(X) + 1e-12))
        cols.append(np.log(np.abs(X) + 1e-12))
        
        # Inverse
        cols.append(1.0 / (1.0 + X**2))
        cols.append(1.0 / (1.0 + np.abs(X)))
        cols.append(1.0 / (X**2 + 0.1))
        
        # Smooth step / Sigmoid
        cols.append(1.0 / (1.0 + np.exp(-X)))
        cols.append(np.tanh(X))
        cols.append(np.tanh(2 * X))
        cols.append(np.tanh(5 * X))
        
        # Sqrt
        cols.append(np.sqrt(np.abs(X) + 1e-12))
        cols.append(np.cbrt(X))
        
        # RBF-like (Gaussian kernels) centered at a few points
        # This helps with localized features
        centers = [-2, -1, -0.5, 0, 0.5, 1, 2]
        widths = [0.5, 1.0, 2.0]
        for c in centers:
            for w in widths:
                cols.append(np.exp(-((X - c)**2) / (2 * w**2)))
                
        # Absolute value
        cols.append(np.abs(X))
        
        # Power functions
        cols.append(np.sign(X) * (np.abs(X) ** 0.5))
        cols.append(np.sign(X) * (np.abs(X) ** 1.5))
        
        return np.column_stack(cols)

    B = get_basis_functions(X)
    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Center and scale y for correlation checks
    y_centered = y - y_mean
    y_norm = np.linalg.norm(y_centered)
    
    if y_norm < 1e-12:
        return lambda X: np.full(len(X), y_mean)

    B_centered = B - np.mean(B, axis=0)
    
    # Strategy 1: Check for single basis function fit
    best_r2 = -np.inf
    best_idx = -1
    
    for j in range(B.shape[1]):
        b_norm = np.linalg.norm(B_centered[:, j])
        if b_norm < 1e-12:
            continue
        corr = np.dot(B_centered[:, j], y_centered) / (b_norm * y_norm)
        r2 = corr * corr
        if r2 > best_r2:
            best_r2 = r2
            best_idx = j
            
    # If a single feature explains > 99.99% of variance, fit it and return
    if best_r2 > 0.9999:
        X_f = B[:, best_idx]
        X_aug = np.column_stack([np.ones(n), X_f])
        w, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        
        def predict_single(X_pred):
            X_pred = np.asarray(X_pred, dtype=np.float64)
            B_pred = get_basis_functions(X_pred)
            B_pred = np.nan_to_num(B_pred, nan=0.0, posinf=0.0, neginf=0.0)
            X_f_pred = B_pred[:, best_idx]
            X_aug_pred = np.column_stack([np.ones(len(X_pred)), X_f_pred])
            return X_aug_pred @ w
            
        return predict_single

    # Strategy 2: Try small combinations of features (2 or 3) if single wasn't enough
    if best_r2 < 0.999:
        # Try all pairs
        best_pair_r2 = best_r2
        best_pair = (best_idx, -1)
        
        # Pre-compute R2 scores to limit search
        r2_scores = []
        for j in range(B.shape[1]):
            b_norm = np.linalg.norm(B_centered[:, j])
            if b_norm < 1e-12:
                r2_scores.append((0, j))
            else:
                corr = np.dot(B_centered[:, j], y_centered) / (b_norm * y_norm)
                r2_scores.append((corr*corr, j))
        
        r2_scores.sort(reverse=True)
        top_k = 20
        top_indices = [idx for _, idx in r2_scores[:top_k]]
        
        for i in range(len(top_indices)):
            for j in range(i+1, len(top_indices)):
                idx1, idx2 = top_indices[i], top_indices[j]
                X_pair = np.column_stack([np.ones(n), B[:, idx1], B[:, idx2]])
                try:
                    w_pair, *_ = np.linalg.lstsq(X_pair, y, rcond=None)
                    y_hat = X_pair @ w_pair
                    sse = np.sum((y - y_hat)**2)
                    r2_pair = 1 - sse / np.sum(y_centered**2)
                    if r2_pair > best_pair_r2:
                        best_pair_r2 = r2_pair
                        best_pair = (idx1, idx2)
                except:
                    pass
                if time_left() < 5.0:
                    break
            if time_left() < 5.0:
                break
        
        if best_pair_r2 > 0.9999:
            idx1, idx2 = best_pair
            X_f = np.column_stack([np.ones(n), B[:, idx1], B[:, idx2]])
            w, *_ = np.linalg.lstsq(X_f, y, rcond=None)
            
            def predict_pair(X_pred):
                X_pred = np.asarray(X_pred, dtype=np.float64)
                B_pred = get_basis_functions(X_pred)
                B_pred = np.nan_to_num(B_pred, nan=0.0, posinf=0.0, neginf=0.0)
                X_f_pred = np.column_stack([np.ones(len(X_pred)), B_pred[:, idx1], B_pred[:, idx2]])
                return X_f_pred @ w
                
            return predict_pair

    # Strategy 3: Generalized Least Squares with all features, with regularization
    # Use Ridge Regression to handle collinearity.
    # Standardize features for stability.
    
    B_mean = np.mean(B, axis=0)
    B_std = np.std(B, axis=0)
    B_std[B_std < 1e-12] = 1.0
    
    B_stdized = (B - B_mean) / B_std
    X_ridge = np.column_stack([np.ones(n), B_stdized])
    
    best_alpha = 1e-6
    best_w = None
    best_sse = np.inf
    
    # Try a range of alphas
    alphas = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0]
    
    for alpha in alphas:
        try:
            A = X_ridge.T @ X_ridge
            A_reg = A.copy()
            A_reg[1:, 1:] += alpha * np.eye(X_ridge.shape[1] - 1)
            b = X_ridge.T @ y
            
            w_ridge = np.linalg.solve(A_reg, b)
            
            y_hat = X_ridge @ w_ridge
            sse = np.sum((y - y_hat)**2)
            
            if sse < best_sse:
                best_sse = sse
                best_w = w_ridge
                best_alpha = alpha
        except:
            continue
        
        if time_left() < 2.0:
            break

    # Define predict using the standardized basis
    def predict_general(X_pred):
        X_pred = np.asarray(X_pred, dtype=np.float64)
        B_pred = get_basis_functions(X_pred)
        B_pred = np.nan_to_num(B_pred, nan=0.0, posinf=0.0, neginf=0.0)
        B_pred_std = (B_pred - B_mean) / B_std
        X_pred_ridge = np.column_stack([np.ones(len(X_pred)), B_pred_std])
        return X_pred_ridge @ best_w

    # Final check: is the general fit actually good?
    # If R2 is terrible, fall back to simple polynomial degree 2
    y_hat_final = predict_general(X)
    sse_final = np.sum((y - y_hat_final)**2)
    r2_final = 1 - sse_final / np.sum(y_centered**2)
    
    if r2_final < 0.5:
        def features_simple(X):
            cols = [np.ones(len(X))] + [X[:, j] for j in range(X.shape[1])] + [X[:, i] * X[:, j] for i in range(X.shape[1]) for j in range(i, X.shape[1])]
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features_simple(X), y, rcond=None)
        return lambda X: features_simple(X) @ w

    return predict_general

# EVOLVE-BLOCK-END
