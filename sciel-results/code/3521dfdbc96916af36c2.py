"""Symbolic regression via kernel ridge regression with composite kernel and hyperparameter search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    """Fit a kernel ridge regression model with composite kernel and return predict callable."""
    start_time = time.time()
    deadline = start_time + 280.0  # 50s safety margin from 330s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Normalize inputs to [0, 1] for better kernel behavior
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range < 1e-12] = 1.0
    X_norm = (X - X_min) / X_range
    
    def compute_kernel(X1, X2, params):
        """Compute composite kernel: RBF + polynomial"""
        # RBF kernel
        # Fix broadcasting: X1 shape (n1, d), X2 shape (n2, d)
        # X1[:, None, :] shape (n1, 1, d), X2[None, :, :] shape (1, n2, d)
        # Diff shape (n1, n2, d)
        diff = X1[:, None, :] - X2[None, :, :]
        sq_dist = np.sum(diff ** 2, axis=2)
        
        l2 = params['l2']
        K_rbf = np.exp(-l2 * sq_dist)
        
        # Polynomial kernel
        deg = params['deg']
        gamma = params['gamma']
        coef0 = params['coef0']
        K_poly = (gamma * (X1 @ X2.T) + coef0) ** deg
        
        # Composite kernel
        w_rbf = params['w_rbf']
        w_poly = params['w_poly']
        
        K = w_rbf * K_rbf + w_poly * K_poly
        return K
    
    def kernel_ridge_predict(X_new, X_train_norm, y_train, K_train, alpha, params):
        """Predict using kernel ridge regression"""
        X_new_norm = (X_new - X_min) / X_range
        K_test = compute_kernel(X_new_norm, X_train_norm, params)
        return K_test @ alpha
    
    # Hyperparameter grid search
    # Simplified search to avoid timeout and ensure robustness
    best_score = np.inf
    best_params = None
    best_alpha = None
    
    # Define a smaller, more effective grid
    l2_values = [0.5, 1.0, 2.0, 5.0]
    deg_values = [1, 2]
    gamma_values = [0.5, 1.0]
    coef0_values = [1.0]
    alpha_values = [1e-4, 1e-3, 1e-2, 0.1]
    
    candidate_params = []
    
    # RBF-only
    for l2 in l2_values:
        for alpha in alpha_values:
            candidate_params.append({
                'l2': l2, 'deg': 1, 'gamma': 1.0, 'coef0': 1.0,
                'w_rbf': 1.0, 'w_poly': 0.0, 'alpha': alpha
            })
    
    # Polynomial-only
    for deg in deg_values:
        for gamma in gamma_values:
            for alpha in alpha_values:
                candidate_params.append({
                    'l2': 1.0, 'deg': deg, 'gamma': gamma, 'coef0': 1.0,
                    'w_rbf': 0.0, 'w_poly': 1.0, 'alpha': alpha
                })
    
    # Mixed RBF + Polynomial
    for l2 in [1.0, 2.0]:
        for deg in [2]:
            for gamma in [1.0]:
                for w_rbf in [0.5, 1.0]:
                    for w_poly in [0.5, 1.0]:
                        for alpha in [1e-3, 1e-2]:
                            candidate_params.append({
                                'l2': l2, 'deg': deg, 'gamma': gamma, 'coef0': 1.0,
                                'w_rbf': w_rbf, 'w_poly': w_poly, 'alpha': alpha
                            })
    
    # Evaluate candidates
    n_candidates = len(candidate_params)
    print(f"Trying {n_candidates} kernel configurations...")
    
    for i, params in enumerate(candidate_params):
        if time.time() > deadline - 30:  # Stop if we're close to deadline
            break
        
        alpha = params['alpha']
        K_train = compute_kernel(X_norm, X_norm, params)
        
        # Solve (K + alpha*I) @ alpha_vec = y
        try:
            A = K_train + alpha * np.eye(n)
            alpha_vec = np.linalg.solve(A, y)
            
            # Use a simple cross-validation score (k-fold CV is expensive, use LOO approximation or simple split)
            # For speed, we'll use a 5-fold cross-validation on the training set
            # But since n=200, even simple CV is okay. Let's do a simple 2-fold CV for speed.
            # Split into two halves
            half = n // 2
            idx = np.arange(n)
            np.random.seed(42)
            np.random.shuffle(idx)
            
            # Fold 1
            train_idx = idx[:half]
            val_idx = idx[half:]
            
            K_train_fold = K_train[np.ix_(train_idx, train_idx)]
            y_train_fold = y[train_idx]
            
            A_fold = K_train_fold + alpha * np.eye(len(train_idx))
            alpha_vec_fold = np.linalg.solve(A_fold, y_train_fold)
            
            K_val_fold = K_train[np.ix_(val_idx, train_idx)]
            pred_val_fold = K_val_fold @ alpha_vec_fold
            mse_fold1 = np.mean((pred_val_fold - y[val_idx]) ** 2)
            
            # Fold 2
            train_idx = idx[half:]
            val_idx = idx[:half]
            
            K_train_fold = K_train[np.ix_(train_idx, train_idx)]
            y_train_fold = y[train_idx]
            
            A_fold = K_train_fold + alpha * np.eye(len(train_idx))
            alpha_vec_fold = np.linalg.solve(A_fold, y_train_fold)
            
            K_val_fold = K_train[np.ix_(val_idx, train_idx)]
            pred_val_fold = K_val_fold @ alpha_vec_fold
            mse_fold2 = np.mean((pred_val_fold - y[val_idx]) ** 2)
            
            score = (mse_fold1 + mse_fold2) / 2.0
            
            if score < best_score:
                best_score = score
                best_params = params
                # Re-solve on full training set for final model
                A = K_train + alpha * np.eye(n)
                best_alpha = np.linalg.solve(A, y)
                
        except np.linalg.LinAlgError:
            continue
    
    # If we found a good model, use it. Otherwise, fall back to a simple RBF.
    if best_params is None:
        # Fallback: simple RBF with default params
        params = {'l2': 1.0, 'deg': 1, 'gamma': 1.0, 'coef0': 1.0,
                  'w_rbf': 1.0, 'w_poly': 0.0, 'alpha': 1e-3}
        K_train = compute_kernel(X_norm, X_norm, params)
        A = K_train + params['alpha'] * np.eye(n)
        best_alpha = np.linalg.solve(A, y)
        best_params = params
    
    # Define predict function
    def predict(X_new):
        X_new = np.asarray(X_new, dtype=np.float64)
        X_new_norm = (X_new - X_min) / X_range
        K_test = compute_kernel(X_new_norm, X_norm, best_params)
        return K_test @ best_alpha
    
    return predict
# EVOLVE-BLOCK-END
