"""Sparse Kernel Regression with Nyström approximation for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

def _rbf_kernel(A, B, length_scale):
    """Compute RBF kernel between sets A and B."""
    # Efficient computation: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    A_sq = np.sum(A**2, axis=1)[:, None]
    B_sq = np.sum(B**2, axis=1)[None, :]
    sq_dists = A_sq + B_sq - 2 * (A @ B.T)
    sq_dists = np.maximum(sq_dists, 0.0)
    return np.exp(-sq_dists / (2.0 * length_scale**2))

def _nystrom_approx(K, n_landmarks, rng):
    """Select landmarks and compute Nyström approximation.
    
    Returns:
        W: (n, n_landmarks) matrix such that K_approx = W @ W.T
        landmarks: indices of selected landmarks
    """
    n = K.shape[0]
    if n_landmarks >= n:
        return np.sqrt(np.maximum(K, 0)), np.arange(n)
    
    # Select landmarks using uniform sampling with replacement (simpler, fast)
    # Or use greedy. For simplicity and speed, use random selection.
    landmarks = rng.choice(n, size=n_landmarks, replace=False)
    
    # K_L: (n_landmarks, n_landmarks)
    K_L = K[np.ix_(landmarks, landmarks)]
    # K_L_inv: need to invert K_L + alpha*I for stability
    alpha = 1e-6
    try:
        K_L_inv = np.linalg.inv(K_L + alpha * np.eye(n_landmarks))
    except np.linalg.LinAlgError:
        K_L_inv = np.linalg.pinv(K_L + alpha * np.eye(n_landmarks))
    
    # K_XL: (n, n_landmarks)
    K_XL = K[:, landmarks]
    
    # W = K_XL @ K_L_inv @ K_L^{-1/2} ... actually for prediction we need
    # K(X, X_train) approx = K(X, L) @ K_L^{-1} @ K(L, X_train)
    # Let's store the components for prediction
    # W such that K_approx = W @ W.T
    # W = K_XL @ K_L_inv @ K_L^{1/2} ... this is getting complex.
    # Let's just return the components needed for prediction.
    
    return K_XL, K_L_inv, landmarks

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 320.0  # Safety margin from 330s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Normalize inputs to [0, 1] for better kernel scaling
    x_min = X.min(axis=0, keepdims=True)
    x_max = X.max(axis=0, keepdims=True)
    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0
    X_norm = (X - x_min) / x_range
    
    # Determine number of landmarks
    # For n=200, using all points is feasible, but let's use Nyström for safety
    # Actually, n=200 is small. Full KRR is O(n^3) = 8M ops, very fast.
    # But the directive says to use Sparse Kernel Regression with Nyström.
    # Let's use a subset to be safe and follow the strategy.
    n_landmarks = min(50, n)
    
    # Try a few length scales and regularization strengths
    # We'll do a quick grid search over length_scale and alpha (regularization)
    
    # Compute full kernel for training data (n x n)
    # But we want to use Nyström. Let's select landmarks first.
    rng = np.random.RandomState(42)
    
    # Heuristic for initial length scale: median distance
    dists = cdist(X_norm, X_norm, 'euclidean')
    median_dist = np.median(dists[dists > 0])
    if median_dist == 0:
        median_dist = 1.0
    
    # Grid of length scales around median
    length_scales = [median_dist * 0.25, median_dist * 0.5, median_dist, median_dist * 2.0, median_dist * 4.0]
    # Regularization strengths (elastic net lambda)
    lambdas = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1]
    
    best_score = np.inf
    best_predict = None
    best_params = None
    
    # Pre-compute landmark indices (fixed for all trials)
    landmarks = rng.choice(n, size=n_landmarks, replace=False)
    
    for ls in length_scales:
        if time.time() > DEADLINE:
            break
        # Compute K(X_norm, X_norm) - but we only need K(X, L) and K(L, L)
        K_XL = _rbf_kernel(X_norm, X_norm[landmarks], ls)  # (n, n_landmarks)
        K_LL = _rbf_kernel(X_norm[landmarks], X_norm[landmarks], ls)  # (n_landmarks, n_landmarks)
        
        for lam in lambdas:
            if time.time() > DEADLINE:
                break
            
            # Solve for coefficients: (K_LL + lam * I) @ c = y_landmarks
            # But we want to fit all y, not just y_landmarks.
            # In Nyström KRR, we approximate K ≈ K_XL @ K_LL^{-1} @ K_LX
            # The model is f(x) = K(x, L) @ K_LL^{-1} @ c
            # Where c solves: (K_LL + lam * I) @ c ≈ K_LL^{-1} @ K_XL.T @ y
            # This is equivalent to solving: (K_LL + lam * K_LL^{-1} K_XL.T K_XL) ... no.
            
            # Standard Nyström KRR:
            # Minimize ||y - K_approx @ w||^2 + lam * w.T @ K_LL @ w
            # Where K_approx = K_XL @ K_LL^{-1} @ K_LX
            # The optimal w satisfies: (K_XL @ K_LL^{-1} @ K_LX @ K_XL @ K_LL^{-1} + lam * I) w = K_XL @ K_LL^{-1} @ y
            # This is expensive.
            
            # Alternative: Fit coefficients c in the landmark space directly.
            # f(x) = K(x, L) @ c
            # Minimize ||y - K_XL @ c||^2 + lam * c.T @ K_LL @ c
            # Solution: c = (K_XL.T @ K_XL + lam * K_LL)^{-1} @ K_XL.T @ y
            
            try:
                A = K_XL.T @ K_XL + lam * K_LL
                b = K_XL.T @ y
                c = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            
            # Compute training error (approximate)
            y_pred = K_XL @ c
            mse = np.mean((y - y_pred)**2)
            var_y = np.var(y)
            if var_y == 0:
                norm_mse = mse
            else:
                norm_mse = mse / var_y
            
            if norm_mse < best_score:
                best_score = norm_mse
                best_params = (ls, lam, c.copy())
                # Create predict function
                def predict(X, ls=ls, c=c, landmarks=landmarks, x_min=x_min, x_range=x_range):
                    X = np.asarray(X, dtype=np.float64)
                    X_norm = (X - x_min) / x_range
                    K_XL_test = _rbf_kernel(X_norm, X_norm[landmarks], ls)
                    return K_XL_test @ c
                best_predict = predict
    
    # If no good params found, fall back to simple polynomial
    if best_predict is None:
        # Fallback: degree-2 polynomial
        def features(X):
            X = np.asarray(X, dtype=np.float64)
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            return np.column_stack(cols)
        
        w, *_ = np.linalg.lstsq(features(X), y, rcond=None)
        def predict(X):
            return features(X) @ w
        return predict
    
    return best_predict

# EVOLVE-BLOCK-END
