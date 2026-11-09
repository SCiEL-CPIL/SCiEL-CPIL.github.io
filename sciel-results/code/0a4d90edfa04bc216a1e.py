import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290  # 290s limit to be safe within 300s/330s
    
    X = X_train
    y = y_train
    
    # 1. Normalize X to [0, 1] for better numerical conditioning
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    x_range = x_max - x_min
    # Avoid division by zero
    x_range[x_range == 0] = 1.0
    
    X_norm = (X - x_min) / x_range
    
    # 2. Model A: High-Degree Polynomial with Ridge Regression
    # Degree 12 is usually sufficient for smooth 1D functions
    degree = 12
    def poly_features(X_n, d):
        cols = [np.ones(len(X_n))]
        for i in range(1, d + 1):
            cols.append(X_n[:, 0] ** i)
        return np.column_stack(cols)
    
    Phi_A = poly_features(X_norm, degree)
    # Ridge regression
    lam_A = 1e-5
    try:
        w_A = np.linalg.solve(Phi_A.T @ Phi_A + lam_A * np.eye(degree + 1), Phi_A.T @ y)
        pred_A_train = Phi_A @ w_A
        mse_A = np.mean((pred_A_train - y) ** 2)
    except np.linalg.LinAlgError:
        w_A = np.zeros(degree + 1)
        pred_A_train = np.zeros_like(y)
        mse_A = np.inf

    # 3. Model B: RBF Kernel Regression
    # Use 25 centers evenly spaced in the normalized range [0, 1]
    n_centers = 25
    centers = np.linspace(0, 1, n_centers).reshape(-1, 1)
    
    def rbf_features(X_n, gamma):
        # X_n: (N, 1), centers: (M, 1)
        # dist: (N, M)
        diff = X_n - centers.T
        sq_dist = diff ** 2
        return np.exp(-gamma * sq_dist)
    
    best_mse_B = np.inf
    best_w_B = None
    best_gamma_B = None
    
    # Try a range of gamma values (inverse of bandwidth squared)
    # For normalized data in [0,1], gamma around 5-50 is often good
    gammas = [1, 5, 10, 20, 50, 100, 200]
    
    for gamma in gammas:
        if time.time() > deadline:
            break
        K = rbf_features(X_norm, gamma)
        # Add regularization to handle potential collinearity or ill-conditioning
        lam_B = 1e-4
        try:
            w_B = np.linalg.solve(K.T @ K + lam_B * np.eye(n_centers), K.T @ y)
            pred_B_train = K @ w_B
            mse_B = np.mean((pred_B_train - y) ** 2)
            
            if mse_B < best_mse_B:
                best_mse_B = mse_B
                best_w_B = w_B
                best_gamma_B = gamma
        except np.linalg.LinAlgError:
            continue

    # 4. Select the better model
    # Use a small penalty for complexity or just pick lowest train error
    # Since we have no test data, lowest train error is a reasonable heuristic
    # but RBF can overfit. Ridge Poly is more stable.
    # Let's add a small bias towards the simpler model if errors are close.
    
    if best_w_B is None:
        # Fallback to Poly
        def predict(X):
            X_n = (X - x_min) / x_range
            return poly_features(X_n, degree) @ w_A
        return predict
    
    if mse_A < best_mse_B * 1.1: # Prefer Poly if it's within 10% of RBF
        def predict(X):
            X_n = (X - x_min) / x_range
            return poly_features(X_n, degree) @ w_A
    else:
        def predict(X):
            X_n = (X - x_min) / x_range
            return rbf_features(X_n, best_gamma_B) @ best_w_B

    return predict
