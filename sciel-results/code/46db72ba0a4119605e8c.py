"""Rewritten program: uses LOESS (local polynomial regression) with adaptive bandwidth via GCV."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 285  # Safety margin for 330s limit
    
    X = np.asarray(X_train, dtype=float).reshape(-1, 1)
    y = np.asarray(y_train, dtype=float).flatten()
    n = len(y)
    
    if n < 2:
        mean_y = np.mean(y) if n > 0 else 0.0
        return lambda X: np.full(len(np.asarray(X).reshape(-1)), mean_y)
    
    x_vals = X[:, 0].copy()
    y_vals = y.copy()
    
    # Check if y is constant
    if np.std(y_vals) < 1e-12:
        mean_y = np.mean(y_vals)
        return lambda X: np.full(len(np.asarray(X).reshape(-1)), mean_y)
    
    y_var = np.var(y_vals)
    if y_var < 1e-15:
        mean_y = np.mean(y_vals)
        return lambda X: np.full(len(np.asarray(X).reshape(-1)), mean_y)
    
    # Sort by x for efficient neighborhood finding
    sort_idx = np.argsort(x_vals)
    x_sorted = x_vals[sort_idx]
    y_sorted = y_vals[sort_idx]
    
    # LOESS with local quadratic fit (degree=2)
    # Bandwidth: fraction of points used in local fit
    # We'll try several bandwidths and select via GCV
    
    # Precompute cumulative sums for fast weighted least squares
    # For local polynomial regression, we need weighted moments
    
    def loess_predict(x_query, x_train, y_train, bandwidth, degree=2):
        """
        LOESS prediction for 1D data.
        bandwidth: fraction of data points to use in local fit
        degree: polynomial degree (0, 1, or 2)
        """
        n = len(x_train)
        n_neighbors = max(degree + 2, int(bandwidth * n))
        n_neighbors = min(n_neighbors, n)
        
        x_query = np.asarray(x_query, dtype=float)
        n_query = len(x_query)
        predictions = np.zeros(n_query)
        
        for i in range(n_query):
            if time.time() > deadline - 5:
                # Fallback: use global mean
                predictions[i:] = np.mean(y_train)
                break
            
            xq = x_query[i]
            
            # Find n_neighbors closest points
            # Use searchsorted for efficiency
            idx = np.searchsorted(x_train, xq)
            
            # Build candidate neighborhood
            left = max(0, idx - n_neighbors // 2)
            right = min(n, left + n_neighbors)
            left = max(0, right - n_neighbors)
            
            # Get the actual closest points
            candidates = np.arange(left, right)
            if len(candidates) < n_neighbors:
                # Expand if needed
                extra = n_neighbors - len(candidates)
                if left > 0:
                    new_left = max(0, left - extra)
                    candidates = np.arange(new_left, right)
                else:
                    new_right = min(n, right + extra)
                    candidates = np.arange(left, new_right)
            
            x_local = x_train[candidates]
            y_local = y_train[candidates]
            
            # Compute distances and weights (tricube kernel)
            dist = np.abs(x_local - xq)
            max_dist = np.max(dist) if len(dist) > 0 else 1.0
            if max_dist < 1e-10:
                max_dist = 1.0
            
            # Tricube weights
            u = dist / max_dist
            weights = (1 - u**3)**3
            weights = np.maximum(weights, 1e-10)  # Avoid zero weights
            
            # Weighted local polynomial regression
            # Build design matrix
            # Center x_local at xq for numerical stability
            x_centered = x_local - xq
            
            # Build Vandermonde-like matrix
            if degree == 0:
                A = np.ones((len(x_local), 1))
            elif degree == 1:
                A = np.column_stack([np.ones(len(x_local)), x_centered])
            else:  # degree == 2
                A = np.column_stack([np.ones(len(x_local)), x_centered, x_centered**2])
            
            # Weighted least squares: (A^T W A)^-1 A^T W y
            W_sqrt = np.sqrt(weights)
            A_w = A * W_sqrt[:, None]
            y_w = y_local * W_sqrt
            
            try:
                # Solve using normal equations
                ATA = A_w.T @ A_w
                ATy = A_w.T @ y_w
                
                # Add small regularization for numerical stability
                reg = 1e-8 * np.eye(degree + 1)
                ATA += reg
                
                coeffs = np.linalg.solve(ATA, ATy)
                
                # Predict at xq (which is x_centered = 0)
                # For degree 0: coeffs[0]
                # For degree 1: coeffs[0] + coeffs[1]*0 = coeffs[0]
                # For degree 2: coeffs[0] + coeffs[1]*0 + coeffs[2]*0^2 = coeffs[0]
                predictions[i] = coeffs[0]
                
            except np.linalg.LinAlgError:
                # Fallback: weighted mean
                predictions[i] = np.sum(weights * y_local) / np.sum(weights)
        
        return predictions
    
    def compute_gcv(x_train, y_train, bandwidth, degree=2):
        """
        Compute Generalized Cross-Validation score for LOESS.
        GCV = (1/n) * sum((y_i - y_hat_i)^2) / (1 - h_ii)^2
        where h_ii is the i-th diagonal of the hat matrix.
        """
        n = len(x_train)
        n_neighbors = max(degree + 2, int(bandwidth * n))
        n_neighbors = min(n_neighbors, n)
        
        # For GCV, we need the hat matrix diagonal
        # This is expensive, so we approximate using leave-one-out style
        # Alternative: use a simpler cross-validation approach
        
        # Use k-fold cross-validation as a proxy
        n_folds = 5
        indices = np.arange(n)
        np.random.seed(42)
        indices = np.random.permutation(indices)
        
        mse_sum = 0.0
        for fold in range(n_folds):
            val_idx = indices[fold * n // n_folds : (fold + 1) * n // n_folds]
            train_idx = np.concatenate([indices[:fold * n // n_folds], indices[(fold + 1) * n // n_folds:]])
            
            if len(train_idx) < degree + 2:
                continue
            
            x_tr = x_train[train_idx]
            y_tr = y_train[train_idx]
            x_val = x_train[val_idx]
            y_val = y_train[val_idx]
            
            y_pred = loess_predict(x_val, x_tr, y_tr, bandwidth, degree)
            mse_sum += np.mean((y_val - y_pred)**2)
        
        return mse_sum / n_folds
    
    # Try several bandwidths and select the best via cross-validation
    bandwidths = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    best_bw = 0.3  # Default
    best_cv = np.inf
    
    for bw in bandwidths:
        if time.time() > deadline - 30:
            break
        
        cv_score = compute_gcv(x_sorted, y_sorted, bw, degree=2)
        
        if cv_score < best_cv:
            best_cv = cv_score
            best_bw = bw
    
    # Final LOESS fit with selected bandwidth
    # Precompute predictions for all training points (not needed, but we prepare the predict function)
    
    def predict(X_pred):
        X_pred = np.asarray(X_pred, dtype=float).reshape(-1, 1)[:, 0]
        return loess_predict(X_pred, x_sorted, y_sorted, best_bw, degree=2)
    
    return predict
# EVOLVE-BLOCK-END
