import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 310  # 20s safety margin before 330s kill

    def time_ok():
        return time.time() < deadline

    def build_features(X):
        d = X.shape[1]
        n = len(X)
        feats = [np.ones(n)]
        
        # Linear
        for i in range(d):
            feats.append(X[:, i])
            
        # Quadratic
        for i in range(d):
            for j in range(i, d):
                feats.append(X[:, i] * X[:, j])
                
        # Cubic (only for small d to keep feature count manageable)
        if d <= 3:
            for i in range(d):
                for j in range(i, d):
                    for k in range(j, d):
                        feats.append(X[:, i] * X[:, j] * X[:, k])
        
        # Richer non-linear single vars
        for i in range(d):
            x = X[:, i]
            # Trigonometric
            feats.append(np.sin(x))
            feats.append(np.cos(x))
            feats.append(np.sin(2*x))
            feats.append(np.cos(2*x))
            feats.append(np.sin(0.5*x))
            feats.append(np.cos(0.5*x))
            # Exponential (scaled to avoid overflow)
            feats.append(np.exp(-0.1 * x))
            feats.append(np.exp(-0.2 * x))
            feats.append(np.exp(-0.3 * x))
            feats.append(np.exp(-0.5 * x))
            feats.append(np.exp(-1.0 * x))
            # Logarithmic
            feats.append(np.log(1 + x))
            feats.append(np.log(x))
            # Power functions
            feats.append(np.sqrt(x))
            feats.append(np.cbrt(x))
            feats.append(x**2)  # already have but explicit
            feats.append(np.tanh(x))
            feats.append(1.0 / (1.0 + np.exp(-x)))  # sigmoid
            
        # Cross terms: linear * linear (already in quadratic)
        # Cross terms: non-linear * linear
        for i in range(d):
            for j in range(d):
                if i != j:
                    xi = X[:, i]
                    xj = X[:, j]
                    feats.append(np.sin(xi) * xj)
                    feats.append(np.cos(xi) * xj)
                    feats.append(np.exp(-0.2 * xi) * xj)
                    feats.append(np.log(1 + xi) * xj)
                    feats.append(np.sqrt(xi) * xj)
                    feats.append(np.tanh(xi) * xj)
                    
        # Cross terms: non-linear * non-linear
        for i in range(d):
            for j in range(i+1, d):
                xi = X[:, i]
                xj = X[:, j]
                feats.append(np.sin(xi) * np.sin(xj))
                feats.append(np.cos(xi) * np.cos(xj))
                feats.append(np.sin(xi) * np.cos(xj))
                feats.append(np.exp(-0.2 * xi) * np.exp(-0.2 * xj))
                feats.append(np.log(1 + xi) * np.log(1 + xj))
                feats.append(np.sqrt(xi) * np.sqrt(xj))
                
        # Sum and product of all inputs (for symmetric equations)
        if d >= 2:
            feats.append(np.sum(X, axis=1))
            feats.append(np.prod(X, axis=1))
            if d >= 3:
                feats.append(np.sum(X**2, axis=1))
                
        F = np.column_stack(feats)
        # Handle NaN/Inf
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Standardize
        mean = F.mean(axis=0)
        std = F.std(axis=0)
        std[std < 1e-10] = 1.0
        F_std = (F - mean) / std
        
        return F_std, mean, std

    F_std, mean, std = build_features(X_train)
    p = F_std.shape[1]
    
    # Use OMP (Orthogonal Matching Pursuit) for feature selection + Ridge refinement
    # This is faster than forward selection and handles noiseless data well
    
    # First, try plain least squares as baseline
    try:
        w_lsq, *_ = np.linalg.lstsq(F_std, y_train, rcond=None)
        pred_lsq = F_std @ w_lsq
        mse_lsq = np.mean((pred_lsq - y_train)**2)
    except:
        w_lsq = None
        mse_lsq = float('inf')
    
    # OMP: select k features that best explain y
    def omp_select(F, y, max_k=50, tol=1e-10):
        """Orthogonal Matching Pursuit for feature selection."""
        n, p = F.shape
        residual = y.copy()
        selected = []
        F_transpose = F.T  # (p, n)
        
        # Precompute norms for efficiency
        for k in range(max_k):
            if not time_ok():
                break
            if len(selected) == p:
                break
            
            # Find feature most correlated with residual
            correlations = np.abs(F_transpose @ residual)
            # Mask already selected
            for s in selected:
                correlations[s] = 0
            
            best_feat = np.argmax(correlations)
            if correlations[best_feat] < tol * np.linalg.norm(residual):
                break
                
            selected.append(best_feat)
            
            # Project residual onto selected features
            F_sel = F[:, selected]
            try:
                coeff, *_ = np.linalg.lstsq(F_sel, y, rcond=None)
            except:
                break
            residual = y - F_sel @ coeff
            
            if np.linalg.norm(residual) < tol:
                break
                
        return selected
    
    if time_ok():
        selected_feats = omp_select(F_std, y_train, max_k=80)
    else:
        selected_feats = list(range(min(p, 30)))
    
    # Fit on selected features with small ridge
    F_sel = F_std[:, selected_feats]
    p_sel = len(selected_feats)
    
    best_w = None
    best_mse = float('inf')
    
    # Try several regularization levels
    for lam in [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        if not time_ok():
            break
        try:
            if lam == 0.0:
                w, *_ = np.linalg.lstsq(F_sel, y_train, rcond=None)
            else:
                XTX = F_sel.T @ F_sel
                XTy = F_sel.T @ y_train
                A = XTX + lam * np.eye(p_sel)
                w = np.linalg.solve(A, XTy)
            
            pred = F_sel @ w
            mse = np.mean((pred - y_train)**2)
            
            if mse < best_mse:
                best_mse = mse
                best_w = w
        except:
            continue
    
    # If OMP+Ridge didn't work well, fall back to full lstsq
    if best_w is None or (w_lsq is not None and mse_lsq < best_mse):
        if w_lsq is not None:
            final_w = w_lsq
            use_full = True
        else:
            final_w, *_ = np.linalg.lstsq(F_std, y_train, rcond=None)
            use_full = True
    else:
        use_full = False
        # Create full weight vector from selected
        final_w_full = np.zeros(p)
        final_w_full[selected_feats] = best_w
        final_w = final_w_full

    def predict(X):
        d = X.shape[1]
        n = len(X)
        feats = [np.ones(n)]
        
        for i in range(d):
            feats.append(X[:, i])
        for i in range(d):
            for j in range(i, d):
                feats.append(X[:, i] * X[:, j])
        if d <= 3:
            for i in range(d):
                for j in range(i, d):
                    for k in range(j, d):
                        feats.append(X[:, i] * X[:, j] * X[:, k])
        for i in range(d):
            x = X[:, i]
            feats.append(np.sin(x))
            feats.append(np.cos(x))
            feats.append(np.sin(2*x))
            feats.append(np.cos(2*x))
            feats.append(np.sin(0.5*x))
            feats.append(np.cos(0.5*x))
            feats.append(np.exp(-0.1 * x))
            feats.append(np.exp(-0.2 * x))
            feats.append(np.exp(-0.3 * x))
            feats.append(np.exp(-0.5 * x))
            feats.append(np.exp(-1.0 * x))
            feats.append(np.log(1 + x))
            feats.append(np.log(x))
            feats.append(np.sqrt(x))
            feats.append(np.cbrt(x))
            feats.append(x**2)
            feats.append(np.tanh(x))
            feats.append(1.0 / (1.0 + np.exp(-x)))
        for i in range(d):
            for j in range(d):
                if i != j:
                    xi = X[:, i]
                    xj = X[:, j]
                    feats.append(np.sin(xi) * xj)
                    feats.append(np.cos(xi) * xj)
                    feats.append(np.exp(-0.2 * xi) * xj)
                    feats.append(np.log(1 + xi) * xj)
                    feats.append(np.sqrt(xi) * xj)
                    feats.append(np.tanh(xi) * xj)
        for i in range(d):
            for j in range(i+1, d):
                xi = X[:, i]
                xj = X[:, j]
                feats.append(np.sin(xi) * np.sin(xj))
                feats.append(np.cos(xi) * np.cos(xj))
                feats.append(np.sin(xi) * np.cos(xj))
                feats.append(np.exp(-0.2 * xi) * np.exp(-0.2 * xj))
                feats.append(np.log(1 + xi) * np.log(1 + xj))
                feats.append(np.sqrt(xi) * np.sqrt(xj))
        if d >= 2:
            feats.append(np.sum(X, axis=1))
            feats.append(np.prod(X, axis=1))
            if d >= 3:
                feats.append(np.sum(X**2, axis=1))
                
        F_new = np.column_stack(feats)
        F_new = np.nan_to_num(F_new, nan=0.0, posinf=0.0, neginf=0.0)
        F_new_std = (F_new - mean) / std
        
        preds = F_new_std @ final_w
        preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
        return preds

    return predict
