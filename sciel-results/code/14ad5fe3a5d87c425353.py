"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    t_start = time.time()
    deadline = 290.0  # safety margin before 300s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Normalize inputs for kernel computation
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    def compute_kernel(X1, X2, params):
        """
        Compute composite kernel matrix.
        params: dict with keys 'rbf_width', 'poly_degree', 'poly_gamma', 'poly_coef0', 'periodic_length', 'periodic_gamma'
        """
        # RBF kernel
        X1_sq = np.sum(X1 ** 2, axis=1)
        X2_sq = np.sum(X2 ** 2, axis=1)
        dist_sq = X1_sq[:, None] + X2_sq[None, :] - 2 * X1 @ X2.T
        np.maximum(dist_sq, 0, out=dist_sq)
        K_rbf = np.exp(-dist_sq / (2 * params['rbf_width'] ** 2))
        
        # Polynomial kernel
        K_poly = (params['poly_gamma'] * (X1 @ X2.T) + params['poly_coef0']) ** params['poly_degree']
        
        # Periodic kernel
        # Use average length scale for simplicity
        K_per = np.exp(-2 * np.sum((X1[:, None, :] - X2[None, :, :]) ** 2 / (2 * params['periodic_length'] ** 2), axis=2) * params['periodic_gamma'])
        
        # Composite kernel
        K = K_rbf + K_poly + K_per
        
        return K
    
    def krr_fit(X_train_norm, y_train_norm, params, alpha):
        """
        Fit KRR model with given parameters and regularization.
        """
        K = compute_kernel(X_train_norm, X_train_norm, params)
        n = len(X_train_norm)
        try:
            A = K + alpha * np.eye(n)
            w = np.linalg.solve(A, y_train_norm)
        except:
            try:
                w, *_ = np.linalg.lstsq(K + alpha * np.eye(n), y_train_norm, rcond=None)
            except:
                w = np.zeros(n)
        
        def predict(X_pred):
            X_pred_norm = (X_pred - X_mean) / X_std
            K_pred = compute_kernel(X_pred_norm, X_train_norm, params)
            pred_norm = K_pred @ w
            pred = pred_norm * y_std + y_mean
            return pred
        
        return predict
    
    def cross_validate(X_train_norm, y_train_norm, params, alpha, n_folds=5):
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        fold_size = n // n_folds
        errors = []
        
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
            
            X_train_cv = X_train_norm[train_idx]
            y_train_cv = y_train_norm[train_idx]
            X_val_cv = X_train_norm[val_idx]
            y_val_cv = y_train_norm[val_idx]
            
            K_train = compute_kernel(X_train_cv, X_train_cv, params)
            n_train = len(X_train_cv)
            
            try:
                A = K_train + alpha * np.eye(n_train)
                w = np.linalg.solve(A, y_train_cv)
            except:
                continue
            
            K_val = compute_kernel(X_val_cv, X_train_cv, params)
            pred = K_val @ w
            mse = np.mean((pred - y_val_cv) ** 2)
            errors.append(mse)
        
        if len(errors) == 0:
            return np.inf
        return np.mean(errors)
    
    best_predictor = None
    best_cv_score = np.inf
    
    # Grid search over kernel hyperparameters
    rbf_widths = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    poly_degrees = [1, 2, 3, 4]
    poly_gammas = [0.1, 0.5, 1.0, 2.0]
    poly_coef0s = [0.5, 1.0, 2.0]
    periodic_lengths = [0.5, 1.0, 2.0, 5.0]
    periodic_gammas = [0.5, 1.0, 2.0]
    alphas = [1e-4, 1e-3, 1e-2, 0.1, 1.0]
    
    # To keep search tractable, we'll do a coarse grid and then refine
    # First pass: coarse grid
    for rbf_w in rbf_widths:
        for poly_deg in poly_degrees:
            for poly_g in poly_gammas:
                for poly_c0 in poly_coef0s:
                    for per_l in periodic_lengths:
                        for per_g in periodic_gammas:
                            if time.time() > deadline - 30:
                                break
                            for alpha in alphas:
                                params = {
                                    'rbf_width': rbf_w,
                                    'poly_degree': poly_deg,
                                    'poly_gamma': poly_g,
                                    'poly_coef0': poly_c0,
                                    'periodic_length': per_l,
                                    'periodic_gamma': per_g
                                }
                                try:
                                    cv_score = cross_validate(X_norm, y_norm, params, alpha)
                                    if cv_score < best_cv_score:
                                        best_cv_score = cv_score
                                        best_predictor = krr_fit(X_norm, y_norm, params, alpha)
                                except Exception:
                                    pass
                            if time.time() > deadline - 30:
                                break
                        if time.time() > deadline - 30:
                            break
                    if time.time() > deadline - 30:
                        break
                if time.time() > deadline - 30:
                    break
            if time.time() > deadline - 30:
                break
        if time.time() > deadline - 30:
            break
    
    # Fallback if nothing worked
    if best_predictor is None:
        params = {
            'rbf_width': 1.0,
            'poly_degree': 2,
            'poly_gamma': 0.5,
            'poly_coef0': 1.0,
            'periodic_length': 1.0,
            'periodic_gamma': 1.0
        }
        best_predictor = krr_fit(X_norm, y_norm, params, 0.1)
    
    return best_predictor
# EVOLVE-BLOCK-END
