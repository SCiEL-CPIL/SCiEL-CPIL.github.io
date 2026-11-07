import numpy as np
import time
from scipy.linalg import solve

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 295.0  # Safety margin from 330s limit
    
    n_samples, n_features = X_train.shape
    
    # Helper: generate rich basis features (polynomial + non-polynomial)
    def get_basis_features(X, degree=4):
        cols = [np.ones(len(X))]
        d = X.shape[1]
        
        # Non-polynomial features per dimension
        for i in range(d):
            x = X[:, i]
            cols.append(x)
            cols.append(np.sin(x))
            cols.append(np.cos(x))
            cols.append(np.log(x))
            cols.append(1.0 / x)
            # Scaled exp to avoid overflow (x in [1,10])
            cols.append(np.exp(-0.1 * x))
            cols.append(np.exp(-0.5 * x))
        
        # Degree 2
        for i in range(d):
            for j in range(i, d):
                cols.append(X[:, i] * X[:, j])
        
        # Degree 3
        if degree >= 3:
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
    
    # RBF sigmas to try - broader range
    sigmas = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
    # Ridge lambdas
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 5.0, 10.0, 50.0]
    
    # Pre-compute RBF kernels for each sigma
    rbf_kernels = {}
    for sigma in sigmas:
        rbf_kernels[sigma] = np.exp(-dist_sq / (2.0 * sigma ** 2))
    
    # Pre-compute basis features for full training set
    basis_full = get_basis_features(X_train, degree=4)
    
    # 5-fold cross-validation
    n_folds = 5
    rng = np.random.RandomState(42)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    fold_indices = np.array_split(indices, n_folds)
    
    # Pre-compute basis features for each fold's train and val sets
    fold_data = {}
    for i in range(n_folds):
        val_idx = fold_indices[i]
        train_idx = np.concatenate([fold_indices[j] for j in range(n_folds) if j != i])
        
        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        
        B_tr = get_basis_features(X_tr, degree=4)
        B_val = get_basis_features(X_val, degree=4)
        
        fold_data[i] = {
            'train_idx': train_idx,
            'val_idx': val_idx,
            'X_tr': X_tr,
            'y_tr': y_tr,
            'X_val': X_val,
            'y_val': y_val,
            'B_tr': B_tr,
            'B_val': B_val,
        }
    
    # Collect CV scores for all (sigma, lambda) pairs
    cv_results = []  # list of (score, sigma, lambda)
    
    for sigma in sigmas:
        if time.time() > deadline - 15.0:
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
                B_tr = fd['B_tr']
                B_val = fd['B_val']
                y_tr = fd['y_tr']
                y_val = fd['y_val']
                
                # RBF train-train for this fold
                R_tr = K_sigma[np.ix_(train_idx, train_idx)]
                
                # Combined design matrix for training
                Phi_tr = np.hstack([B_tr, R_tr])
                
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
                Phi_val = np.hstack([B_val, R_val])
                y_pred = Phi_val @ w
                
                mse = np.mean((y_pred - y_val) ** 2)
                cv_errors.append(mse)
            
            mean_cv = np.mean(cv_errors)
            cv_results.append((mean_cv, sigma, lam))
    
    # Sort by CV score and take top-K for ensemble
    cv_results.sort(key=lambda x: x[0])
    
    # Use top-3 models for ensemble (if available), else top-1
    top_k = min(3, len(cv_results))
    best_configs = cv_results[:top_k]
    
    # Fit final models on all data for each top config
    final_models = []
    for _, sigma, lam in best_configs:
        K_full = rbf_kernels[sigma]
        Phi_full = np.hstack([basis_full, K_full])
        
        n_params = Phi_full.shape[1]
        A = Phi_full.T @ Phi_full + lam * np.eye(n_params)
        b = Phi_full.T @ y_train
        
        try:
            w_final = solve(A, b)
        except Exception:
            try:
                w_final = np.linalg.lstsq(A, b, rcond=None)[0]
            except Exception:
                continue
        
        final_models.append((sigma, w_final))
    
    # Capture for prediction
    train_X = X_train
    
    if not final_models:
        # Fallback: simple mean
        mean_y = np.mean(y_train)
        def predict(X):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return np.full(len(X), mean_y)
        return predict
    
    def predict(X):
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Basis features
        B_pred = get_basis_features(X, degree=4)
        
        # RBF features and predictions for each model
        preds = []
        for sigma, w in final_models:
            X_sq = np.sum(X ** 2, axis=1)
            train_sq = np.sum(train_X ** 2, axis=1)
            dist_sq_pred = X_sq[:, None] + train_sq[None, :] - 2.0 * (X @ train_X.T)
            np.maximum(dist_sq_pred, 0, out=dist_sq_pred)
            R_pred = np.exp(-dist_sq_pred / (2.0 * sigma ** 2))
            
            Phi_pred = np.hstack([B_pred, R_pred])
            preds.append(Phi_pred @ w)
        
        # Ensemble: average predictions
        return np.mean(preds, axis=0)
    
    return predict
