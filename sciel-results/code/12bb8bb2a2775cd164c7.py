import numpy as np

def fit(X_train, y_train):
    # Handle edge case where y is constant
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    if y_std < 1e-10:
        return lambda X: np.full(X.shape[0], y_mean)
    
    y_norm = (y_train - y_mean) / y_std

    # Normalize inputs
    mean_x = np.mean(X_train, axis=0)
    std_x = np.std(X_train, axis=0)
    std_x[std_x < 1e-10] = 1.0
    X_norm = (X_train - mean_x) / std_x

    # Hyperparameters
    sigmas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    lambdas = [1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0]
    
    # Simple K-Fold Cross Validation
    n_samples = len(X_norm)
    n_folds = 5
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    best_score = np.inf
    best_sigma = 1.0
    best_lambda = 1.0
    
    for sigma in sigmas:
        gamma = 1.0 / (2.0 * sigma**2)
        for lam in lambdas:
            fold_scores = []
            for i in range(n_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                
                X_tr = X_norm[train_idx]
                y_tr = y_norm[train_idx]
                X_val = X_norm[val_idx]
                y_val = y_norm[val_idx]
                
                # Compute Kernel Matrix for Training
                # K = exp(-gamma * ||x_i - x_j||^2) + x_i . x_j + 1
                # Efficient distance calculation
                x_sq = np.sum(X_tr**2, axis=1)
                dist2 = x_sq[:, None] + x_sq[None, :] - 2 * (X_tr @ X_tr.T)
                dist2 = np.maximum(dist2, 0)
                K_rbf = np.exp(-gamma * dist2)
                K_lin = X_tr @ X_tr.T
                K = K_rbf + K_lin + 1.0
                
                # Regularize and Solve
                A = K + lam * np.eye(len(X_tr))
                try:
                    alpha = np.linalg.solve(A, y_tr)
                except np.linalg.LinAlgError:
                    alpha = np.linalg.lstsq(A, y_tr, rcond=None)[0]
                    
                # Compute Kernel for Validation
                x_sq_val = np.sum(X_val**2, axis=1)
                dist2_val = x_sq_val[:, None] + x_sq[None, :] - 2 * (X_val @ X_tr.T)
                dist2_val = np.maximum(dist2_val, 0)
                K_val_rbf = np.exp(-gamma * dist2_val)
                K_val_lin = X_val @ X_tr.T
                K_val = K_val_rbf + K_val_lin + 1.0
                
                y_pred_val = K_val @ alpha
                mse = np.mean((y_pred_val - y_val)**2)
                fold_scores.append(mse)
                
            mean_score = np.mean(fold_scores)
            if mean_score < best_score:
                best_score = mean_score
                best_sigma = sigma
                best_lambda = lam

    # Final Fit on all data
    gamma = 1.0 / (2.0 * best_sigma**2)
    x_sq = np.sum(X_norm**2, axis=1)
    dist2 = x_sq[:, None] + x_sq[None, :] - 2 * (X_norm @ X_norm.T)
    dist2 = np.maximum(dist2, 0)
    K_rbf = np.exp(-gamma * dist2)
    K_lin = X_norm @ X_norm.T
    K = K_rbf + K_lin + 1.0
    
    A = K + best_lambda * np.eye(len(X_norm))
    try:
        alpha = np.linalg.solve(A, y_norm)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(A, y_norm, rcond=None)[0]

    def predict(X):
        X_norm_test = (X - mean_x) / std_x
        x_sq_test = np.sum(X_norm_test**2, axis=1)
        dist2_test = x_sq_test[:, None] + x_sq[None, :] - 2 * (X_norm_test @ X_norm.T)
        dist2_test = np.maximum(dist2_test, 0)
        K_test_rbf = np.exp(-gamma * dist2_test)
        K_test_lin = X_norm_test @ X_norm.T
        K_test = K_test_rbf + K_test_lin + 1.0
        y_pred_norm = K_test @ alpha
        return y_pred_norm * y_std + y_mean

    return predict
