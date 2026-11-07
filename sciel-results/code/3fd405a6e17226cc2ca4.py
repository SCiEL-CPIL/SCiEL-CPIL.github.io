import numpy as np
import time
from itertools import product

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 30.0  # Safety margin per equation, total 300s for 10
    
    n_samples, n_features = X_train.shape
    
    # 1. Generate Polynomial Features (Degree 1 to 3)
    # This captures algebraic structures like x*y, x^2, etc.
    def get_poly_features(X, degree=3):
        cols = [np.ones(len(X))]
        # Degree 1
        for i in range(X.shape[1]):
            cols.append(X[:, i])
        # Degree 2
        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                cols.append(X[:, i] * X[:, j])
        # Degree 3
        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                for k in range(j, X.shape[1]):
                    cols.append(X[:, i] * X[:, j] * X[:, k])
        return np.column_stack(cols)

    poly_feat = get_poly_features(X_train, degree=3)
    
    # 2. Generate RBF Features
    # We use all training points as centers to maximize flexibility.
    # K(x, z) = exp(-||x-z||^2 / (2*sigma^2))
    
    # Pre-compute squared distances between all pairs in X_train
    # dist^2 = ||x||^2 + ||z||^2 - 2*x.z
    X_norm_sq = np.sum(X_train**2, axis=1)
    dist_sq = X_norm_sq[:, None] + X_norm_sq[None, :] - 2 * (X_train @ X_train.T)
    np.maximum(dist_sq, 0, out=dist_sq) # Numerical stability
    
    # Grid search parameters
    # Sigma controls the width of the RBF. 
    # Input ranges are [1,3], [1,2], [1,5], [1,10]. Distances can be up to ~10-20.
    # Small sigma -> local fit, large sigma -> global smooth fit.
    sigmas = [0.5, 1.0, 2.0, 5.0, 10.0]
    lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    
    # Cross-Validation Setup
    n_folds = 5
    indices = np.arange(n_samples)
    np.random.seed(42) # Fixed seed for reproducibility in CV split
    np.random.shuffle(indices)
    fold_indices = np.array_split(indices, n_folds)
    
    best_score = np.inf
    best_params = None
    best_w = None
    
    # Pre-compute RBF kernels for all sigmas to avoid repeated exp calculations in inner loop if possible?
    # Actually, exp is fast. 200x200 matrix.
    rbf_kernels = {}
    for sigma in sigmas:
        # K_ii is always 1.
        # We need K for training set to solve Ridge.
        # For prediction, we need K_new.
        # For CV, we need K for training subset and validation subset.
        # Let's just compute full K_train for each sigma.
        rbf_kernels[sigma] = np.exp(-dist_sq / (2 * sigma**2))
        
    # Evaluate combinations
    for sigma in sigmas:
        if time.time() > deadline - 5.0:
            break
        K_train = rbf_kernels[sigma]
        
        for lam in lambdas:
            if time.time() > deadline - 2.0:
                break
                
            cv_errors = []
            for i in range(n_folds):
                val_idx = fold_indices[i]
                train_idx = np.concatenate([fold_indices[j] for j in range(n_folds) if j != i])
                
                X_tr = X_train[train_idx]
                y_tr = y_train[train_idx]
                X_val = X_train[val_idx]
                y_val = y_train[val_idx]
                
                # Build Design Matrix for Training
                # Poly features for train
                P_tr = get_poly_features(X_tr, degree=3)
                # RBF features for train (subset of K_train)
                R_tr = K_train[np.ix_(train_idx, train_idx)]
                
                # Combined features
                Phi_tr = np.hstack([P_tr, R_tr])
                
                # Solve Ridge Regression: (Phi^T Phi + lam I) w = Phi^T y
                # Adding small lambda to identity to ensure invertibility
                A = Phi_tr.T @ Phi_tr + lam * np.eye(Phi_tr.shape[1])
                b = Phi_tr.T @ y_tr
                
                try:
                    w = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    w = np.linalg.lstsq(A, b, rcond=None)[0]
                    
                # Predict Validation
                P_val = get_poly_features(X_val, degree=3)
                # RBF features for validation against training centers
                # dist_val_sq = ||x_val||^2 + ||x_tr||^2 - 2 x_val . x_tr
                X_val_norm_sq = np.sum(X_val**2, axis=1)
                X_tr_norm_sq = np.sum(X_tr**2, axis=1)
                dist_val_sq = X_val_norm_sq[:, None] + X_tr_norm_sq[None, :] - 2 * (X_val @ X_tr.T)
                np.maximum(dist_val_sq, 0, out=dist_val_sq)
                R_val = np.exp(-dist_val_sq / (2 * sigma**2))
                
                Phi_val = np.hstack([P_val, R_val])
                y_pred = Phi_val @ w
                
                mse = np.mean((y_pred - y_val)**2)
                cv_errors.append(mse)
            
            mean_cv_error = np.mean(cv_errors)
            
            if mean_cv_error < best_score:
                best_score = mean_cv_error
                best_params = (sigma, lam)
                # We don't save w here, we re-solve on full data later
                # But we can save the indices to speed up final solve if needed, 
                # though 200x200 solve is trivial.

    # Final Fit with Best Parameters
    sigma, lam = best_params
    
    # Build Full Design Matrix
    P_full = get_poly_features(X_train, degree=3)
    R_full = rbf_kernels[sigma]
    Phi_full = np.hstack([P_full, R_full])
    
    A = Phi_full.T @ Phi_full + lam * np.eye(Phi_full.shape[1])
    b = Phi_full.T @ y_train
    
    try:
        w_final = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w_final = np.linalg.lstsq(A, b, rcond=None)[0]
        
    # Capture parameters for prediction
    final_sigma = sigma
    final_lam = lam
    final_w = w_final
    train_X = X_train

    def predict(X):
        # Generate Poly Features
        P_pred = get_poly_features(X, degree=3)
        
        # Generate RBF Features
        # Distances between X (new) and train_X (centers)
        X_norm_sq = np.sum(X**2, axis=1)
        train_X_norm_sq = np.sum(train_X**2, axis=1)
        dist_sq = X_norm_sq[:, None] + train_X_norm_sq[None, :] - 2 * (X @ train_X.T)
        np.maximum(dist_sq, 0, out=dist_sq)
        R_pred = np.exp(-dist_sq / (2 * final_sigma**2))
        
        Phi_pred = np.hstack([P_pred, R_pred])
        
        return Phi_pred @ final_w

    return predict
