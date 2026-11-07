import numpy as np
import time
from scipy.linalg import solve

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin from 300s limit
    
    n_samples, n_features = X_train.shape
    
    # Helper: generate polynomial features up to given degree
    def get_poly_features(X, degree=4):
        cols = [np.ones(len(X))]
        d = X.shape[1]
        # Degree 1
        for i in range(d):
            cols.append(X[:, i])
        # Degree 2
        for i in range(d):
            for j in range(i, d):
                cols.append(X[:, i] * X[:, j])
        # Degree 3
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    cols.append(X[:, i] * X[:, j] * X[:, k])
        # Degree 4
        if degree >= 4:
            for i in range(d):
                for j in range(i, d):
                    for k in range(j, d):
                        for l in range(k, d):
                            cols.append(X[:, i] * X[:, j] * X[:, k] * X[:, l])
        return np.column_stack(cols)
    
    # Pre-compute squared norms for RBF distance calculations
    X_sq_norms = np.sum(X_train ** 2, axis=1)
    
    # Pre-compute all pairwise squared distances
    dist_sq = X_sq_norms[:, None] + X_sq_norms[None, :] - 2.0 * (X_train @ X_train.T)
    np.maximum(dist_sq, 0, out=dist_sq)
    
    # RBF sigmas to try (on raw scale, since input ranges are [1,3], [1,2], [1,5], [1,10])
    sigmas = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    # Ridge lambdas
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 5.0, 10.0]
    
    # Pre-compute RBF kernels for each sigma
    rbf_kernels = {}
    for sigma in sigmas:
        rbf_kernels[sigma] = np.exp(-dist_sq / (2.0 * sigma ** 2))
    
    # Pre-compute polynomial features for full training set
    poly_full = get_poly_features(X_train, degree=4)
    
    # 5-fold cross-validation
    n_folds = 5
    rng = np.random.RandomState(42)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    fold_indices = np.array_split(indices, n_folds)
    
    best_score = np.inf
    best_sigma = sigmas[len(sigmas) // 2]
    best_lambda = lambdas[len(lambdas) // 2]
    
    # Pre-compute poly features for each fold's train and val sets
    # This is expensive but n=200 is small
    fold_data = {}
    for i in range(n_folds):
        val_idx = fold_indices[i]
        train_idx = np.concatenate([fold_indices[j] for j in range(n_folds) if j != i])
        
        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        
        P_tr = get_poly_features(X_tr, degree=4)
        P_val = get_poly_features(X_val, degree=4)
        
        # RBF: train-train
        R_tr = rbf_kernels[best_sigma][np.ix_(train_idx, train_idx)]  # placeholder, will recompute per sigma
        # Actually we need per-sigma, so we'll compute inside the loop
        
        fold_data[i] = {
            'train_idx': train_idx,
            'val_idx': val_idx,
            'X_tr': X_tr,
            'y_tr': y_tr,
            'X_val': X_val,
            'y_val': y_val,
            'P_tr': P_tr,
            'P_val': P_val,
        }
    
    # Search over sigma and lambda
    for sigma in sigmas:
        if time.time() > deadline - 10.0:
            break
        
        K_sigma = rbf_kernels[sigma]
        
        for lam in lambdas:
            if time.time() > deadline - 5.0:
                break
            
            cv_errors = []
            for i in range(n_folds):
                fd = fold_data[i]
                train_idx = fd['train_idx']
                val_idx = fd['val_idx']
                P_tr = fd['P_tr']
                P_val = fd['P_val']
                y_tr = fd['y_tr']
                y_val = fd['y_val']
                
                # RBF train-train for this fold
                R_tr = K_sigma[np.ix_(train_idx, train_idx)]
                
                # Combined design matrix for training
                Phi_tr = np.hstack([P_tr, R_tr])
                
                # Ridge solve
                n_params = Phi_tr.shape[1]
                A = Phi_tr.T @ Phi_tr + lam * np.eye(n_params)
                b = Phi_tr.T @ y_tr
                
                try:
                    w = solve(A, b)
                except Exception:
                    try:
                        w = np.linalg.lstsq(A, b, rcond=None)[0]
                    except Exception:
                        cv_errors.append(np.inf)
                        continue
                
                # RBF val-train for this fold
                X_val = fd['X_val']
                X_tr_centers = X_train[train_idx]
                X_val_sq = np.sum(X_val ** 2, axis=1)
                X_tr_sq = np.sum(X_tr_centers ** 2, axis=1)
                dist_val_sq = X_val_sq[:, None] + X_tr_sq[None, :] - 2.0 * (X_val @ X_tr_centers.T)
                np.maximum(dist_val_sq, 0, out=dist_val_sq)
                R_val = np.exp(-dist_val_sq / (2.0 * sigma ** 2))
                
                # Combined prediction features
                Phi_val = np.hstack([P_val, R_val])
                y_pred = Phi_val @ w
                
                mse = np.mean((y_pred - y_val) ** 2)
                cv_errors.append(mse)
            
            mean_cv = np.mean(cv_errors)
            if mean_cv < best_score:
                best_score = mean_cv
                best_sigma = sigma
                best_lambda = lam
    
    # Final fit on all data with best parameters
    sigma = best_sigma
    lam = best_lambda
    
    K_full = rbf_kernels[sigma]
    Phi_full = np.hstack([poly_full, K_full])
    
    n_params = Phi_full.shape[1]
    A = Phi_full.T @ Phi_full + lam * np.eye(n_params)
    b = Phi_full.T @ y_train
    
    try:
        w_final = solve(A, b)
    except Exception:
        try:
            w_final = np.linalg.lstsq(A, b, rcond=None)[0]
        except Exception:
            # Fallback: simple mean
            mean_y = np.mean(y_train)
            def predict(X):
                return np.full(len(X), mean_y)
            return predict
    
    # Capture for prediction
    train_X = X_train
    final_sigma = sigma
    final_w = w_final
    final_poly_degree = 4
    
    def predict(X):
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Poly features
        P_pred = get_poly_features(X, degree=final_poly_degree)
        
        # RBF features: distances between X and train_X
        X_sq = np.sum(X ** 2, axis=1)
        train_sq = np.sum(train_X ** 2, axis=1)
        dist_sq_pred = X_sq[:, None] + train_sq[None, :] - 2.0 * (X @ train_X.T)
        np.maximum(dist_sq_pred, 0, out=dist_sq_pred)
        R_pred = np.exp(-dist_sq_pred / (2.0 * final_sigma ** 2))
        
        Phi_pred = np.hstack([P_pred, R_pred])
        return Phi_pred @ final_w
    
    return predict
