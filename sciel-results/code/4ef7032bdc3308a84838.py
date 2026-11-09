"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def fit(X_train, y_train):
    start_time = time.time()
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    
    # Fallback: constant
    def predict_const(X):
        return np.full(len(X), y_mean)
    
    if y_var < 1e-12:
        return predict_const

    # Helper to check time
    def time_left():
        return 280.0 - (time.time() - start_time)  # 300s total, 10 eqs -> 30s each, use 28s to be safe? No, 300s total for ALL 10. So 30s per equation. Let's use 25s per equation to be safe.
    
    # Actually, the limit is 300s for ALL 10 equations.
    # So we have 30s per equation on average.
    # Let's set a hard limit per fit call to 25 seconds.
    MAX_TIME_PER_EQ = 25.0
    
    def check_time():
        return (time.time() - start_time) < MAX_TIME_PER_EQ

    # Feature Generation Strategies
    # We want to capture: polynomials, trig, exp, log, interactions.
    
    # 1. Polynomial Features (up to degree 3)
    def gen_poly_features(X, max_deg=3):
        feats = [np.ones(len(X))]
        # Linear
        for j in range(d):
            feats.append(X[:, j])
        # Quadratic
        for i in range(d):
            for j in range(i, d):
                feats.append(X[:, i] * X[:, j])
        # Cubic (only if d is small, e.g., d<=3)
        if d <= 3:
            for i in range(d):
                for j in range(i, d):
                    for k in range(j, d):
                        feats.append(X[:, i] * X[:, j] * X[:, k])
        return np.column_stack(feats)

    # 2. Trigonometric Features
    def gen_trig_features(X):
        feats = [np.ones(len(X))]
        for j in range(d):
            feats.append(np.sin(X[:, j]))
            feats.append(np.cos(X[:, j]))
            # Maybe sin/cos of sums?
            if d > 1:
                for k in range(j+1, d):
                    feats.append(np.sin(X[:, j] + X[:, k]))
                    feats.append(np.cos(X[:, j] - X[:, k]))
        return np.column_stack(feats)

    # 3. Exponential/Log Features
    def gen_explog_features(X):
        feats = [np.ones(len(X))]
        for j in range(d):
            # Clip to avoid overflow
            x_clipped = np.clip(X[:, j], 1e-6, 10.0)
            feats.append(np.exp(x_clipped))
            feats.append(np.log(x_clipped))
            feats.append(1.0 / x_clipped)
        return np.column_stack(feats)

    # 4. Combined Basis
    # Let's try a rich basis: Poly(2) + Trig + Exp/Log
    def gen_combined_features(X):
        f1 = gen_poly_features(X, max_deg=2)
        f2 = gen_trig_features(X)
        f3 = gen_explog_features(X)
        return np.column_stack([f1, f2, f3])

    # 5. Radial Basis Functions (RBF)
    def gen_rbf_features(X, centers=None, sigma=None):
        if centers is None:
            # Use a subset of training data as centers
            idx = np.random.choice(n, size=min(20, n), replace=False)
            centers = X_train[idx]
        if sigma is None:
            # Heuristic sigma
            dists = np.linalg.norm(X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :], axis=-1)
            sigma = np.median(dists[dists > 0]) / 2.0 if np.any(dists > 0) else 1.0
        
        diffs = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=-1)
        return np.exp(-sq_dists / (2.0 * sigma ** 2))

    # Candidates to try
    candidates = []
    
    # 1. Linear
    def feat_linear(X):
        return np.column_stack([np.ones(len(X)), X])
    candidates.append(feat_linear)
    
    # 2. Poly 2
    def feat_poly2(X):
        return gen_poly_features(X, max_deg=2)
    candidates.append(feat_poly2)
    
    # 3. Poly 3 (if d is small)
    if d <= 3:
        def feat_poly3(X):
            return gen_poly_features(X, max_deg=3)
        candidates.append(feat_poly3)
        
    # 4. Trig
    def feat_trig(X):
        return gen_trig_features(X)
    candidates.append(feat_trig)
    
    # 5. Exp/Log
    def feat_explog(X):
        return gen_explog_features(X)
    candidates.append(feat_explog)
    
    # 6. Combined
    def feat_combined(X):
        return gen_combined_features(X)
    candidates.append(feat_combined)
    
    # 7. RBF (fixed centers from train)
    # Pre-compute RBF features for train to check validity
    # RBF can be ill-conditioned, so we use Ridge regression for it
    def feat_rbf(X):
        return gen_rbf_features(X)
    candidates.append(feat_rbf)

    best_model = None
    best_score = np.inf
    best_predict = predict_const

    # Evaluate each candidate
    for i, feat_func in enumerate(candidates):
        if not check_time():
            break
            
        try:
            Xf_train = feat_func(X_train)
            # Check for NaN/Inf
            if not np.all(np.isfinite(Xf_train)):
                continue
                
            # Normalize features for stability?
            # Let's use Ridge regression (L2 regularization) to handle collinearity
            # We can use numpy's lstsq with a small regularization added to diagonal
            # Or just use scipy if available, but let's stick to numpy for speed/simplicity
            # Ridge: min ||Xw - y||^2 + alpha * ||w||^2
            # (X^T X + alpha I) w = X^T y
            
            # Try a few alpha values
            best_alpha_score = np.inf
            best_w = None
            
            for alpha in [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]:
                if not check_time():
                    break
                try:
                    XfT = Xf_train.T
                    A = XfT @ Xf_train + alpha * np.eye(Xf_train.shape[1])
                    b = XfT @ y_train
                    w = np.linalg.solve(A, b)
                    
                    y_pred_train = Xf_train @ w
                    mse_train = np.mean((y_pred_train - y_train) ** 2)
                    norm_mse = mse_train / y_var
                    
                    if norm_mse < best_alpha_score:
                        best_alpha_score = norm_mse
                        best_w = w
                except np.linalg.LinAlgError:
                    continue

            if best_w is not None:
                # Create predict function
                def make_predict(feat_func, w):
                    def predict(X):
                        Xf = feat_func(X)
                        return Xf @ w
                    return predict
                
                pred_func = make_predict(feat_func, best_w)
                
                # We don't have test data, so we rely on train score + regularization
                # Or we can do a quick 5-fold CV? Might be too slow.
                # Let's just pick the one with best regularized train error.
                # The regularization term penalizes complexity, so this should be decent.
                
                if best_alpha_score < best_score:
                    best_score = best_alpha_score
                    best_predict = pred_func
                    best_model = (feat_func, best_w)
                    
        except Exception:
            continue

    # If we have time, try to refine the best model or try more complex combinations?
    # For now, return the best found.
    if best_model is not None:
        return best_predict
    else:
        return predict_const

# EVOLVE-BLOCK-END
