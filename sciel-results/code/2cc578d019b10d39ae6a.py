"""Symbolic regression via adaptive model selection between rich linear basis and Kernel Ridge Regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280  # Safety margin for 4 equations
    
    X = X_train
    y = y_train
    n = len(y)
    
    # Split for validation
    indices = np.arange(n)
    np.random.shuffle(indices)
    split = int(0.8 * n)
    idx_train = indices[:split]
    idx_val = indices[split:]
    
    X_tr = X[idx_train]
    y_tr = y[idx_train]
    X_val = X[idx_val]
    y_val = y[idx_val]
    
    best_val_mse = np.inf
    best_model_params = None
    best_model_type = None
    
    # --- Model 1: Rich Linear Basis with Ridge Regression ---
    def get_linear_features(X):
        x = X[:, 0]
        cols = []
        # Polynomials up to degree 6
        for i in range(7):
            cols.append(x**i)
        # Trigonometric
        for k in [1, 2, 3]:
            cols.append(np.sin(k * x))
            cols.append(np.cos(k * x))
        # Exponentials
        cols.append(np.exp(x))
        cols.append(np.exp(-x))
        # Other non-linear
        cols.append(np.tanh(x))
        cols.append(np.arctan(x))
        return np.column_stack(cols)
    
    F_tr = get_linear_features(X_tr)
    F_val = get_linear_features(X_val)
    
    alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    for alpha in alphas:
        # Ridge: w = (F^T F + alpha I)^-1 F^T y
        # Using lstsq is safer for ill-conditioned
        # Or solve normal equations with regularization
        try:
            A = F_tr.T @ F_tr + alpha * np.eye(F_tr.shape[1])
            b = F_tr.T @ y_tr
            w = np.linalg.solve(A, b)
            
            y_pred_val = F_val @ w
            mse = np.mean((y_pred_val - y_val)**2)
            
            if mse < best_val_mse:
                best_val_mse = mse
                best_model_params = w
                best_model_type = 'linear'
        except np.linalg.LinAlgError:
            pass

    # --- Model 2: Kernel Ridge Regression (RBF) ---
    def rbf_kernel(X1, X2, gamma):
        # X1, X2 are (N, 1)
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
        np.maximum(sq_dists, 0, out=sq_dists) # numerical stability
        return np.exp(-gamma * sq_dists)
    
    gammas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    lambdas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    
    for gamma in gammas:
        K_tr = rbf_kernel(X_tr, X_tr, gamma)
        for lam in lambdas:
            try:
                A = K_tr + lam * np.eye(len(X_tr))
                b = y_tr
                alpha_vec = np.linalg.solve(A, b)
                
                K_val = rbf_kernel(X_val, X_tr, gamma)
                y_pred_val = K_val @ alpha_vec
                mse = np.mean((y_pred_val - y_val)**2)
                
                if mse < best_val_mse:
                    best_val_mse = mse
                    best_model_params = (gamma, alpha_vec, X_tr, y_tr)
                    best_model_type = 'krr'
            except np.linalg.LinAlgError:
                pass

    # Refit on all data with best parameters
    if best_model_type == 'linear':
        F_all = get_linear_features(X)
        alpha = 1e-4 # Default, but we need the specific alpha that gave best val
        # We didn't store the best alpha for linear, just the weights.
        # To be precise, we should store the best alpha. Let's re-run the selection logic to store it.
        # For simplicity and speed, if linear won, we refit with the alpha that was best.
        # Let's just store the best alpha in the loop above.
        
        # Since I didn't store alpha, I will just pick a reasonable one or re-run.
        # Actually, let's just use the weights from the best val split? No, that's biased.
        # Let's assume the best alpha was one of the list. We can just pick the one that minimizes val error again on the full set? No.
        # Let's just re-do the linear part storing the alpha.
        
        best_val_mse_lin = np.inf
        best_w_lin = None
        best_alpha_lin = 1e-4
        for alpha in alphas:
            try:
                A = F_tr.T @ F_tr + alpha * np.eye(F_tr.shape[1])
                b = F_tr.T @ y_tr
                w = np.linalg.solve(A, b)
                y_pred_val = F_val @ w
                mse = np.mean((y_pred_val - y_val)**2)
                if mse < best_val_mse_lin:
                    best_val_mse_lin = mse
                    best_w_lin = w
                    best_alpha_lin = alpha
            except:
                pass
        
        if best_model_type == 'linear' and best_w_lin is not None:
            # Refit on all data
            F_all = get_linear_features(X)
            A = F_all.T @ F_all + best_alpha_lin * np.eye(F_all.shape[1])
            b = F_all.T @ y
            w_final = np.linalg.solve(A, b)
            
            def predict(X_pred):
                return get_linear_features(X_pred) @ w_final
        
        else:
            # Fallback to KRR if linear failed or KRR was better
            if best_model_type == 'krr':
                gamma, _, _, _ = best_model_params
                # We need to find the best lambda for this gamma? 
                # The stored params are from the val split. 
                # Let's just use the stored alpha_vec and gamma? 
                # No, we must refit on all data.
                
                # Find best lambda for the best gamma
                best_val_mse_krr = np.inf
                best_alpha_krr = None
                best_lam_krr = 1e-2
                K_tr = rbf_kernel(X_tr, X_tr, gamma)
                for lam in lambdas:
                    try:
                        A = K_tr + lam * np.eye(len(X_tr))
                        alpha_vec = np.linalg.solve(A, y_tr)
                        K_val = rbf_kernel(X_val, X_tr, gamma)
                        y_pred_val = K_val @ alpha_vec
                        mse = np.mean((y_pred_val - y_val)**2)
                        if mse < best_val_mse_krr:
                            best_val_mse_krr = mse
                            best_alpha_krr = alpha_vec
                            best_lam_krr = lam
                    except:
                        pass
                
                if best_alpha_krr is not None:
                    # Refit on all data
                    K_all = rbf_kernel(X, X, gamma)
                    A = K_all + best_lam_krr * np.eye(len(X))
                    alpha_final = np.linalg.solve(A, y)
                    
                    def predict(X_pred):
                        K_pred = rbf_kernel(X_pred, X, gamma)
                        return K_pred @ alpha_final
                else:
                    # Fallback to simple polynomial
                    def predict(X_pred):
                        return np.zeros(len(X_pred))
            else:
                # Total fallback
                def predict(X_pred):
                    return np.zeros(len(X_pred))
                    
    else:
        # KRR was better
        gamma, _, _, _ = best_model_params
        best_val_mse_krr = np.inf
        best_alpha_krr = None
        best_lam_krr = 1e-2
        K_tr = rbf_kernel(X_tr, X_tr, gamma)
        for lam in lambdas:
            try:
                A = K_tr + lam * np.eye(len(X_tr))
                alpha_vec = np.linalg.solve(A, y_tr)
                K_val = rbf_kernel(X_val, X_tr, gamma)
                y_pred_val = K_val @ alpha_vec
                mse = np.mean((y_pred_val - y_val)**2)
                if mse < best_val_mse_krr:
                    best_val_mse_krr = mse
                    best_alpha_krr = alpha_vec
                    best_lam_krr = lam
            except:
                pass
        
        if best_alpha_krr is not None:
            K_all = rbf_kernel(X, X, gamma)
            A = K_all + best_lam_krr * np.eye(len(X))
            alpha_final = np.linalg.solve(A, y)
            
            def predict(X_pred):
                K_pred = rbf_kernel(X_pred, X, gamma)
                return K_pred @ alpha_final
        else:
            def predict(X_pred):
                return np.zeros(len(X_pred))

    return predict

# EVOLVE-BLOCK-END
