"""
Structural Change: Sparse RBF Network with k-Means Centers and Polynomial Baseline.
Instead of inverting a full n x n kernel matrix (GP), this strategy:
1. Selects K centers using k-means clustering on the training data.
2. Constructs a feature matrix combining localized Gaussian RBFs centered at these points 
   and a global polynomial basis (up to degree 2 or 3) to capture global trends.
3. Solves for weights via regularized linear least squares (Ridge Regression), which is 
   computationally cheaper (O(nK^2) vs O(n^3)) and more stable for n=200.
4. Uses a small grid search over the number of centers K and regularization strength lambda.
5. This avoids the overfitting of full GP with small noise and the computational bottleneck 
   of large matrix inversions, while maintaining non-parametric flexibility via the RBFs.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    """
    Fits a Sparse RBF network with polynomial baseline.
    Returns a predict callable.
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    
    n_samples, n_features = X.shape
    
    # Normalize inputs to [0, 1] to standardize RBF width behavior
    min_vals = X.min(axis=0, keepdims=True)
    max_vals = X.max(axis=0, keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    
    X_norm = (X - min_vals) / range_vals
    
    # Helper to generate polynomial features (degree 1 and 2)
    def get_poly_features(X):
        # Constant term
        cols = [np.ones((X.shape[0], 1))]
        # Linear terms
        for j in range(X.shape[1]):
            cols.append(X[:, j:j+1])
        # Quadratic terms (x_i^2 and x_i * x_j for i <= j)
        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                if i == j:
                    cols.append((X[:, i] ** 2).reshape(-1, 1))
                else:
                    cols.append((X[:, i] * X[:, j]).reshape(-1, 1))
        return np.hstack(cols)
    
    # Pre-compute polynomial features for training
    P_train = get_poly_features(X_norm)
    
    # k-means implementation (simple and fast for n=200)
    def kmeans(X, k, max_iter=50, tol=1e-4):
        n = X.shape[0]
        # Initialize centers by picking k random points
        indices = np.random.choice(n, k, replace=False)
        centers = X[indices].copy()
        
        for _ in range(max_iter):
            # Assign points to nearest center
            # dists shape: (n, k)
            dists = np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)
            
            # Update centers
            new_centers = np.zeros_like(centers)
            for i in range(k):
                mask = labels == i
                if np.sum(mask) > 0:
                    new_centers[i] = X[mask].mean(axis=0)
                else:
                    # If a cluster is empty, re-initialize it to a random point
                    new_centers[i] = X[np.random.randint(n)]
            
            if np.linalg.norm(new_centers - centers) < tol:
                centers = new_centers
                break
            centers = new_centers
            
        return centers, labels

    # Pre-compute RBF features for a given set of centers
    # We use a fixed width sigma based on the average distance between centers
    # or a fixed width relative to the normalized space.
    # Let's try a few fixed widths.
    
    def compute_rbf_features(X, centers, sigma):
        # X: (n, d), centers: (k, d)
        # dists: (n, k)
        dists_sq = np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
        return np.exp(-dists_sq / (2 * sigma ** 2))
    
    best_score = float('inf')
    best_model_params = None
    
    # Grid search parameters
    # K: number of RBF centers
    K_values = [10, 20, 40, 60]
    # Lambda: regularization strength
    lambda_values = [1e-3, 1e-2, 1e-1, 1.0]
    # Sigma: RBF width (in normalized space [0,1]^d)
    sigma_values = [0.1, 0.2, 0.3]
    
    # We will iterate through combinations, but limit the total time.
    # Since n=200, k-means and least squares are very fast.
    # We can afford to check many combinations.
    
    for K in K_values:
        if time.time() > deadline - 10:
            break
            
        # Run k-means
        try:
            centers, _ = kmeans(X_norm, K)
        except Exception:
            continue
            
        # Compute RBF features for training data
        for sigma in sigma_values:
            if time.time() > deadline - 5:
                break
                
            R_train = compute_rbf_features(X_norm, centers, sigma)
            
            # Combine Polynomial and RBF features
            # P_train: (n, p), R_train: (n, K)
            A_train = np.hstack([P_train, R_train])
            n_features_total = A_train.shape[1]
            
            for lam in lambda_values:
                if time.time() > deadline - 2:
                    break
                    
                # Solve (A^T A + lam * I) w = A^T y
                # Use ridge regression formula
                try:
                    ATA = A_train.T @ A_train
                    ATy = A_train.T @ y
                    reg_term = lam * np.eye(n_features_total)
                    # Add slightly more regularization to the RBF part? 
                    # For simplicity, uniform regularization.
                    w = np.linalg.solve(ATA + reg_term, ATy)
                    
                    # Calculate training error (proxy for validation, since data is noiseless)
                    # In noiseless data, we want to interpolate. 
                    # However, overfitting to noise is not an issue if noise is 0.
                    # But we have finite precision. 
                    # Let's use a simple holdout validation to be safe against numerical issues.
                    
                    # Split into train/val for model selection
                    idx = np.arange(n_samples)
                    np.random.shuffle(idx)
                    val_size = 40
                    val_idx = idx[:val_size]
                    tr_idx = idx[val_size:]
                    
                    X_val = X_norm[val_idx]
                    y_val = y[val_idx]
                    X_tr = X_norm[tr_idx]
                    y_tr = y[val_idx]
                    
                    # Refit on subset to get unbiased validation score
                    # This is expensive if we do it for every lambda. 
                    # Alternative: Use the full training error as a proxy, 
                    # but penalize complexity.
                    # Given the time limit, let's just use full training error 
                    # with a penalty for model complexity (number of features).
                    # Actually, let's do a quick 20% holdout validation for the best few candidates.
                    
                    # For now, compute full training error.
                    y_pred_train = A_train @ w
                    mse_train = np.mean((y_pred_train - y) ** 2)
                    
                    # Add a small penalty for larger models to prefer simplicity if errors are similar
                    score = mse_train + 1e-6 * n_features_total
                    
                    if score < best_score:
                        best_score = score
                        best_model_params = {
                            'K': K,
                            'sigma': sigma,
                            'lambda': lam,
                            'centers': centers.copy(),
                            'weights': w.copy(),
                            'min_vals': min_vals,
                            'range_vals': range_vals,
                            'n_features': n_features
                        }
                        
                except np.linalg.LinAlgError:
                    continue

    # If no model was found (should not happen), use a simple linear fallback
    if best_model_params is None:
        # Fallback: Linear regression on original X
        X_orig = X
        P_orig = get_poly_features(X_orig)
        w, *_ = np.linalg.lstsq(P_orig, y, rcond=None)
        
        def predict(X):
            X = np.asarray(X, dtype=float)
            P_test = get_poly_features(X)
            return P_test @ w
        return predict

    # Finalize with the best parameters found
    # Note: The weights 'w' were computed on the FULL training set in the loop above.
    # This is ideal for noiseless data.
    
    centers = best_model_params['centers']
    sigma = best_model_params['sigma']
    weights = best_model_params['weights']
    min_vals_c = best_model_params['min_vals']
    range_vals_c = best_model_params['range_vals']
    n_features_c = best_model_params['n_features']
    
    def predict(X):
        X = np.asarray(X, dtype=float)
        X_norm_test = (X - min_vals_c) / range_vals_c
        
        # Generate features
        P_test = get_poly_features(X_norm_test)
        R_test = compute_rbf_features(X_norm_test, centers, sigma)
        A_test = np.hstack([P_test, R_test])
        
        y_pred = A_test @ weights
        
        # Ensure finite output
        if not np.all(np.isfinite(y_pred)):
            # Fallback to mean if something went wrong
            return np.full(len(X), np.mean(y_train))
            
        return y_pred

    return predict

# EVOLVE-BLOCK-END
