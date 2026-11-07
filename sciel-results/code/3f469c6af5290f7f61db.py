import numpy as np
import time
import sys

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290.0  # Leave 40s buffer for evaluation overhead
    
    X = np.array(X_train, dtype=float)
    y = np.array(y_train, dtype=float)
    n, d = X.shape
    
    # Helper to check time
    def time_ok():
        return time.time() < deadline - 5.0
    
    # 1. Generate Basis
    features = [np.ones(n)]
    
    # Univariate features
    for j in range(d):
        x = X[:, j]
        # Polynomials
        features.extend([x, x**2, x**3, x**4, x**5])
        # Inverses
        features.extend([1.0/x, 1.0/(x**2)])
        # Trig
        features.extend([np.sin(x), np.cos(x), np.sin(2*x), np.cos(2*x)])
        # Hyperbolic
        features.extend([np.tanh(x), np.tanh(2*x)])
        # Exp/Log (clipped to avoid inf)
        features.append(np.exp(np.clip(x, -50, 50)))
        features.append(np.log(np.abs(x) + 1e-6))
        features.append(np.abs(x))
        
    # Bivariate interaction features
    if d > 1:
        for i in range(d):
            for j in range(i+1, d):
                xi = X[:, i]
                xj = X[:, j]
                # Products
                features.extend([
                    xi * xj,
                    xi**2 * xj,
                    xi * xj**2,
                    xi**2 * xj**2
                ])
                # Sums/Diffs
                features.extend([
                    xi + xj,
                    xi - xj
                ])
                # Trig on sums/diffs
                features.extend([
                    np.sin(xi + xj),
                    np.cos(xi + xj),
                    np.tanh(xi + xj),
                    np.sin(xi - xj),
                    np.cos(xi - xj)
                ])
                # Mixed trig/product
                features.extend([
                    np.sin(xi) * np.cos(xj),
                    np.cos(xi) * np.sin(xj),
                    np.tanh(xi) * np.tanh(xj)
                ])
                
    # Trivariate interaction features (if d >= 3)
    if d > 2:
        # We only have 3 dims max in this problem based on description
        # If d == 3:
        if d == 3:
            x0, x1, x2 = X[:, 0], X[:, 1], X[:, 2]
            # Products
            features.extend([
                x0 * x1 * x2,
                x0**2 * x1 * x2,
                x0 * x1**2 * x2,
                x0 * x1 * x2**2
            ])
            # Sums
            s = x0 + x1 + x2
            features.extend([
                s,
                np.sin(s),
                np.cos(s),
                np.tanh(s)
            ])
            
    B = np.column_stack(features)
    # Clean up inf/nan
    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
    
    num_features = B.shape[1]
    
    # 2. Feature Screening
    # Calculate R^2 for each feature individually (plus intercept)
    # This is fast: O(M * N)
    # We want to keep top K features
    K = 80  # Number of candidates to keep for forward selection
    
    # Center y for faster correlation? No, just use lstsq on [1, x_j]
    # Actually, simple correlation is good enough for screening
    y_mean = np.mean(y)
    y_var = np.var(y)
    if y_var == 0:
        # Constant y
        w = np.array([np.mean(y)])
        # Find index of intercept in B
        # We need to return predict that matches this
        # For constant y, we can just return the mean
        def predict_const(X_test):
            X_test = np.array(X_test, dtype=float)
            return np.full(X_test.shape[0], np.mean(y))
        return predict_const

    # Compute correlation of each column with y
    # B is N x M, y is N
    # Correlation = cov(B, y) / (std(B)*std(y))
    # Since we just need ranking, we can use squared correlation
    # Let's use np.corrcoef or manual
    # Manual is faster for many columns
    B_mean = np.mean(B, axis=0)
    B_std = np.std(B, axis=0)
    # Avoid division by zero
    B_std[B_std == 0] = 1.0
    
    B_centered = B - B_mean
    y_centered = y - y_mean
    
    # Correlation squared
    corr_sq = np.sum(B_centered * y_centered, axis=0) ** 2 / (np.sum(B_centered**2, axis=0) * np.sum(y_centered**2))
    
    # Sort features by correlation squared
    # Exclude intercept (index 0) from ranking, we always keep it
    indices = np.argsort(corr_sq)[::-1]
    # Keep top K features, ensuring we don't pick index 0 multiple times
    # Actually, index 0 is intercept. We want top K non-intercept features.
    candidate_indices = []
    for idx in indices:
        if idx == 0:
            continue
        candidate_indices.append(idx)
        if len(candidate_indices) >= K:
            break
    
    # If we have fewer than K features, take all
    if len(candidate_indices) < K:
        candidate_indices = [i for i in range(num_features) if i != 0]
        
    # 3. Forward Selection
    # Start with intercept
    selected_indices = [0]
    remaining_indices = candidate_indices.copy()
    
    # Precompute B.T @ y
    Bt_y = B.T @ y
    
    # We will use Ridge Regression for stability during selection
    # lambda_val = 1e-4
    # To speed up, we can just use OLS for selection and Ridge for final fit
    # OLS is faster if we use QR or Cholesky, but for small k, lstsq is fine
    
    best_rss = np.sum((y - np.mean(y))**2) # RSS with intercept only
    max_steps = 20
    
    # Optimization: Pre-compute B columns for candidates? 
    # No, we need to select from `remaining_indices`.
    
    for step in range(max_steps):
        if not time_ok():
            break
        if not remaining_indices:
            break
            
        best_idx = -1
        best_new_rss = best_rss
        
        # Try each remaining candidate
        # To speed this up, we can vectorize the RSS calculation?
        # RSS for adding column j to model M:
        # Can be computed via Frisch-Waugh-Lovell theorem
        # Residualize y on current model, residualize candidate on current model
        # RSS_new = RSS_old - (corr^2 * RSS_candidate_residual)
        
        # Let's use the FWL theorem for speed
        # 1. Residualize y w.r.t selected_indices
        # 2. Residualize candidate w.r.t selected_indices
        # 3. Calculate reduction in RSS
        
        B_sel = B[:, selected_indices]
        # Solve for residuals of y
        # y_res = y - B_sel @ w_sel
        # We can compute w_sel once outside the loop? No, selected_indices changes.
        
        # Actually, computing residuals for y is O(N*K).
        # Computing residuals for each candidate is O(N*K).
        # Total O(N*K*M_candidates).
        
        # Let's just do direct lstsq for simplicity first, but with a tight loop.
        # If it's too slow, we switch to FWL.
        
        # Let's try FWL:
        # Residualize y
        try:
            w_sel, *_ = np.linalg.lstsq(B_sel, y, rcond=None)
            y_res = y - B_sel @ w_sel
        except np.linalg.LinAlgError:
            break
            
        for idx in remaining_indices:
            if not time_ok():
                break
            x_res = B[:, idx]
            # Residualize x_res w.r.t B_sel
            try:
                w_x, *_ = np.linalg.lstsq(B_sel, x_res, rcond=None)
                x_res_res = x_res - B_sel @ w_x
            except np.linalg.LinAlgError:
                continue
                
            # Calculate new RSS
            # RSS_new = RSS_old - ||proj of y_res on x_res_res||^2
            # proj = (y_res . x_res_res) / ||x_res_res||^2 * x_res_res
            # ||proj||^2 = (y_res . x_res_res)^2 / ||x_res_res||^2
            dot = np.dot(y_res, x_res_res)
            norm_sq = np.dot(x_res_res, x_res_res)
            if norm_sq < 1e-10:
                continue
            reduction = (dot * dot) / norm_sq
            new_rss = best_rss - reduction
            
            if new_rss < best_new_rss:
                best_new_rss = new_rss
                best_idx = idx
                
        if best_idx == -1:
            break
            
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        best_rss = best_new_rss
        
    # 4. Final Fit with Ridge
    B_sel = B[:, selected_indices]
    lambda_val = 1e-4
    I = np.eye(B_sel.shape[1])
    try:
        w = np.linalg.solve(B_sel.T @ B_sel + lambda_val * I, B_sel.T @ y)
    except np.linalg.LinAlgError:
        # Fallback to least squares
        w, *_ = np.linalg.lstsq(B_sel, y, rcond=None)
        
    # 5. Define Predict
    def predict(X_test):
        X_test = np.array(X_test, dtype=float)
        n_test = X_test.shape[0]
        
        # Reconstruct basis
        feats = [np.ones(n_test)]
        
        for j in range(d):
            x = X_test[:, j]
            feats.extend([x, x**2, x**3, x**4, x**5])
            feats.extend([1.0/x, 1.0/(x**2)])
            feats.extend([np.sin(x), np.cos(x), np.sin(2*x), np.cos(2*x)])
            feats.extend([np.tanh(x), np.tanh(2*x)])
            feats.append(np.exp(np.clip(x, -50, 50)))
            feats.append(np.log(np.abs(x) + 1e-6))
            feats.append(np.abs(x))
            
        if d > 1:
            for i in range(d):
                for j in range(i+1, d):
                    xi = X_test[:, i]
                    xj = X_test[:, j]
                    feats.extend([
                        xi * xj,
                        xi**2 * xj,
                        xi * xj**2,
                        xi**2 * xj**2
                    ])
                    feats.extend([
                        xi + xj,
                        xi - xj
                    ])
                    feats.extend([
                        np.sin(xi + xj),
                        np.cos(xi + xj),
                        np.tanh(xi + xj),
                        np.sin(xi - xj),
                        np.cos(xi - xj)
                    ])
                    feats.extend([
                        np.sin(xi) * np.cos(xj),
                        np.cos(xi) * np.sin(xj),
                        np.tanh(xi) * np.tanh(xj)
                    ])
                    
        if d > 2:
            if d == 3:
                x0, x1, x2 = X_test[:, 0], X_test[:, 1], X_test[:, 2]
                feats.extend([
                    x0 * x1 * x2,
                    x0**2 * x1 * x2,
                    x0 * x1**2 * x2,
                    x0 * x1 * x2**2
                ])
                s = x0 + x1 + x2
                feats.extend([
                    s,
                    np.sin(s),
                    np.cos(s),
                    np.tanh(s)
                ])

        B_test = np.column_stack(feats)
        B_test = np.nan_to_num(B_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        B_test_sel = B_test[:, selected_indices]
        
        return B_test_sel @ w

    return predict

