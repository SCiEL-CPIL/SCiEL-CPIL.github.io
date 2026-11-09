import numpy as np
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    X = X_train[:, 0]
    y = y_train
    n = len(X)
    
    # 1. Try Rich Basis LS
    def make_basis(x):
        cols = [np.ones(len(x))]
        x_safe = np.clip(x, -10, 10)
        for k in range(1, 6):
            cols.append(x_safe ** k)
        for freq in [0.5, 1.0, 2.0, 3.0, 4.0]:
            cols.append(np.sin(freq * x_safe))
            cols.append(np.cos(freq * x_safe))
        cols.append(np.exp(x_safe))
        cols.append(np.exp(-x_safe))
        cols.append(np.exp(x_safe ** 2))
        cols.append(np.exp(-x_safe ** 2))
        cols.append(np.sin(x_safe ** 2))
        cols.append(np.cos(x_safe ** 2))
        cols.append(np.log(np.abs(x_safe) + 1e-6))
        cols.append(np.tanh(x_safe))
        cols.append(np.arctan(x_safe))
        return np.column_stack(cols)
    
    A = make_basis(X)
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    y_pred = A @ w
    mse = np.mean((y_pred - y) ** 2)
    var_y = np.var(y)
    nmse = mse / var_y if var_y > 0 else 0
    
    if nmse < 1e-4:
        def predict(X):
            x = X[:, 0]
            return make_basis(x) @ w
        return predict
        
    # 2. Fallback: Kernel Ridge Regression
    # Use Gaussian Kernel
    # K(x, x') = exp(-gamma * ||x - x'||^2)
    # Optimize gamma and alpha (regularization)
    
    # Simple CV or Grid Search
    best_nmse = np.inf
    best_params = None
    best_w = None
    
    # Precompute distances for training
    # X is 1D, so distance is abs(x_i - x_j)
    dists = np.abs(X[:, None] - X[None, :])
    
    gammas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    
    for gamma in gammas:
        K = np.exp(-gamma * dists ** 2)
        for alpha in alphas:
            # Solve (K + alpha*I) w = y
            try:
                w_krr = np.linalg.solve(K + alpha * np.eye(n), y)
                y_pred_krr = K @ w_krr
                mse_krr = np.mean((y_pred_krr - y) ** 2)
                nmse_krr = mse_krr / var_y if var_y > 0 else 0
                if nmse_krr < best_nmse:
                    best_nmse = nmse_krr
                    best_params = (gamma, alpha)
                    best_w = w_krr
            except np.linalg.LinAlgError:
                continue
                
    if best_w is not None and best_nmse < 1e-2:
        gamma, alpha = best_params
        def predict(X):
            x = X[:, 0]
            # Compute kernel between test and train
            dists_test = np.abs(x[:, None] - X[None, :])
            K_test = np.exp(-gamma * dists_test ** 2)
            return K_test @ best_w
        return predict

    # 3. If all else fails, return the LS prediction
    def predict(X):
        x = X[:, 0]
        return make_basis(x) @ w
    return predict
