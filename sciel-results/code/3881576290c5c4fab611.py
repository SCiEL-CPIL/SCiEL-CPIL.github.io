# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.linalg import solve
from scipy.optimize import differential_evolution
import itertools

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290.0  # 40s margin for 330s kill
    
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    if y_var < 1e-12:
        return lambda X: np.full(len(X), y_mean)
    
    def check_time():
        return time.time() < deadline

    # Normalize features to [0, 1] for better kernel scaling
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    X_norm = (X_train - X_min) / X_range
    
    # Normalize y
    y_std = np.std(y_train)
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y_train - y_mean) / y_std

    best_predict = None
    best_cv_mse = np.inf
    
    def safe_evaluate(model, X, y):
        try:
            preds = model(X)
            preds = np.asarray(preds, dtype=float)
            if preds.shape != y.shape:
                return np.inf
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                return np.inf
            mse = np.mean((preds - y) ** 2)
            if np.isnan(mse) or np.isinf(mse):
                return np.inf
            return mse
        except Exception:
            return np.inf

    def update_best(model, cv_mse):
        nonlocal best_predict, best_cv_mse
        if model is None:
            return
        if cv_mse < best_cv_mse:
            best_cv_mse = cv_mse
            # We also verify on train to ensure it's not broken, 
            # but CV is the primary selector.
            # If CV is good, train should be okay.
            best_predict = model

    # Kernel Definitions
    # K: (n, n) matrix
    
    def kernel_poly(X1, X2, deg, c=1.0):
        # (X1 . X2 + c) ^ deg
        dot = X1 @ X2.T + c
        return dot ** deg

    def kernel_rbf(X1, X2, gamma):
        sq_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1)[np.newaxis, :] - 2 * (X1 @ X2.T)
        sq_dists = np.maximum(sq_dists, 0)
        return np.exp(-gamma * sq_dists)

    def kernel_fourier(X1, X2, max_freq):
        # Sum of sines kernel approximation via feature map
        # K(x, x') = sum_{k=1}^{max_freq} [cos(2pi k x) cos(2pi k x') + sin(2pi k x) sin(2pi k x')]
        # This is equivalent to a feature map Phi(x)
        # We will construct the feature matrix directly for speed in linear solve
        # But for LOO CV, we need the kernel matrix or the feature matrix.
        # Let's return the feature matrix for Fourier to use linear regression directly.
        pass # Handled separately

    def get_fourier_features(X, max_freq):
        # X is (n, d)
        # Returns (n, 1 + d*2*max_freq)
        cols = [np.ones(len(X))]
        for j in range(X.shape[1]):
            for k in range(1, max_freq + 1):
                cols.append(np.sin(2 * np.pi * k * X[:, j]))
                cols.append(np.cos(2 * np.pi * k * X[:, j]))
        return np.column_stack(cols)

    def get_poly_features(X, deg):
        # Polynomial features up to degree deg
        cols = [np.ones(len(X))]
        for j in range(X.shape[1]):
            for k in range(1, deg + 1):
                cols.append(X[:, j] ** k)
        # Interactions
        if X.shape[1] >= 2:
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    if deg >= 2:
                        cols.append(X[:, i] * X[:, j])
                    if deg >= 3:
                        for k in range(j, X.shape[1]):
                            cols.append(X[:, i] * X[:, j] * X[:, k])
        return np.column_stack(cols)

    def get_rbf_features(X, centers, gamma):
        # RBF features: exp(-gamma * ||x - c||^2)
        diffs = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        return np.exp(-gamma * dists)

    # Candidate Models
    # We will evaluate each model using LOO-CV
    # For Linear Models (Poly, Fourier), LOO is easy:
    # pred_i = y_i - (y_i - w^T x_i) / (1 + w^T (X^T X + lambda I)^-1 x_i) ? 
    # Actually, for Ridge Regression y = X w + eps, w = (X^T X + lambda I)^-1 X^T y
    # LOO prediction for i: y_hat_i = y_i - (x_i^T (X^T X + lambda I)^-1 x_i) / (1 + x_i^T (X^T X + lambda I)^-1 x_i) * (y_i - y_hat_i_unreg)? 
    # Simpler: Compute H matrix (Hat matrix). 
    # For Ridge: H = X (X^T X + lambda I)^-1 X^T
    # LOO_i = (y_i - (H y)_i) / (1 - H_ii)
    
    def compute_loo_mse_ridge(X, y, lam):
        # X: (n, p)
        # y: (n,)
        n = X.shape[0]
        # Solve (X^T X + lam I) w = X^T y
        A = X.T @ X + lam * np.eye(X.shape[1])
        try:
            w = solve(A, X.T @ y)
        except:
            return np.inf
        y_hat = X @ w
        H = X @ np.linalg.inv(A) @ X.T # This is expensive O(n^3) but n=200 is fine. 
        # Actually, H_ii is the leverage.
        # H = X (X^T X + lam I)^-1 X^T
        # Let's compute H more efficiently or just use the formula:
        # H_ii = x_i^T (X^T X + lam I)^-1 x_i
        # We already have inv(A) if we computed it, but we used solve.
        # Let's re-solve to get the inverse or use the identity.
        # Since n is small, let's just compute H directly for simplicity and robustness.
        try:
            inv_A = np.linalg.inv(A)
            H = X @ inv_A @ X.T
            diag_H = np.diag(H)
            # Avoid division by zero
            denom = 1 - diag_H
            denom[abs(denom) < 1e-12] = 1e-12
            loo_errors = (y - y_hat) / denom
            return np.mean(loo_errors ** 2)
        except:
            return np.inf

    def compute_loo_mse_kernel(K, y, lam):
        # K: (n, n)
        # y: (n,)
        # Model: y = K w + eps, w = (K + lam I)^-1 y
        # Hat matrix H = K (K + lam I)^-1
        # H = I - lam (K + lam I)^-1
        try:
            A = K + lam * np.eye(K.shape[0])
            inv_A = np.linalg.inv(A)
            H = np.eye(K.shape[0]) - lam * inv_A
            diag_H = np.diag(H)
            y_hat = K @ (inv_A @ y)
            denom = 1 - diag_H
            denom[abs(denom) < 1e-12] = 1e-12
            loo_errors = (y - y_hat) / denom
            return np.mean(loo_errors ** 2)
        except:
            return np.inf

    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    
    # 1. Polynomial Features
    for deg in [2, 3, 4]:
        if not check_time(): break
        try:
            F = get_poly_features(X_norm, deg)
            for lam in lambdas:
                cv_mse = compute_loo_mse_ridge(F, y_norm, lam)
                if cv_mse < best_cv_mse:
                    # Store the model parameters
                    A = F.T @ F + lam * np.eye(F.shape[1])
                    w = solve(A, F.T @ y_norm)
                    def predict_poly(X, F_fn, w, y_mean, y_std, deg):
                        F_test = F_fn((X - X_min) / X_range, deg)
                        return y_mean + y_std * (F_test @ w)
                    update_best(lambda X, w=w, deg=deg: predict_poly(X, get_poly_features, w, y_mean, y_std, deg), cv_mse)
        except:
            pass

    # 2. Fourier Features (Sum of Sines)
    for freq in [2, 3, 4, 5, 6]:
        if not check_time(): break
        try:
            F = get_fourier_features(X_norm, freq)
            for lam in lambdas:
                cv_mse = compute_loo_mse_ridge(F, y_norm, lam)
                if cv_mse < best_cv_mse:
                    A = F.T @ F + lam * np.eye(F.shape[1])
                    w = solve(A, F.T @ y_norm)
                    def predict_fourier(X, F_fn, w, y_mean, y_std, freq):
                        F_test = F_fn((X - X_min) / X_range, freq)
                        return y_mean + y_std * (F_test @ w)
                    update_best(lambda X, w=w, freq=freq: predict_fourier(X, get_fourier_features, w, y_mean, y_std, freq), cv_mse)
        except:
            pass

    # 3. RBF Kernel (Gaussian)
    # Use a subset of points as centers to keep it manageable, or use all points.
    # With N=200, using all points is fine.
    centers = X_norm
    for gamma in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        if not check_time(): break
        try:
            K = kernel_rbf(centers, centers, gamma)
            for lam in lambdas:
                cv_mse = compute_loo_mse_kernel(K, y_norm, lam)
                if cv_mse < best_cv_mse:
                    A = K + lam * np.eye(n)
                    w = solve(A, y_norm)
                    def predict_rbf(X, centers, gamma, w, y_mean, y_std):
                        Xn = (X - X_min) / X_range
                        K_test = kernel_rbf(Xn, centers, gamma)
                        return y_mean + y_std * (K_test @ w)
                    update_best(lambda X, w=w, gamma=gamma: predict_rbf(X, centers, gamma, w, y_mean, y_std), cv_mse)
        except:
            pass

    # 4. Polynomial Kernel
    for deg in [2, 3, 4]:
        if not check_time(): break
        try:
            K = kernel_poly(X_norm, X_norm, deg, c=1.0)
            for lam in lambdas:
                cv_mse = compute_loo_mse_kernel(K, y_norm, lam)
                if cv_mse < best_cv_mse:
                    A = K + lam * np.eye(n)
                    w = solve(A, y_norm)
                    def predict_poly_kernel(X, deg, c, w, y_mean, y_std):
                        Xn = (X - X_min) / X_range
                        K_test = kernel_poly(Xn, X_norm, deg, c)
                        return y_mean + y_std * (K_test @ w)
                    update_best(lambda X, w=w, deg=deg: predict_poly_kernel(X, deg, 1.0, w, y_mean, y_std), cv_mse)
        except:
            pass

    # 5. Mixed: Linear + RBF
    # This is often very effective.
    # Feature matrix: [1, X, RBF(X, X)]
    # We can approximate this by adding a linear term to the RBF kernel.
    # K_mix = K_rbf + K_lin
    for gamma in [1.0, 5.0, 10.0]:
        if not check_time(): break
        try:
            K_rbf = kernel_rbf(X_norm, X_norm, gamma)
            K_lin = X_norm @ X_norm.T
            K_mix = K_rbf + K_lin
            for lam in lambdas:
                cv_mse = compute_loo_mse_kernel(K_mix, y_norm, lam)
                if cv_mse < best_cv_mse:
                    A = K_mix + lam * np.eye(n)
                    w = solve(A, y_norm)
                    def predict_mix(X, gamma, w, y_mean, y_std):
                        Xn = (X - X_min) / X_range
                        K_test_rbf = kernel_rbf(Xn, X_norm, gamma)
                        K_test_lin = Xn @ X_norm.T
                        K_test = K_test_rbf + K_test_lin
                        return y_mean + y_std * (K_test @ w)
                    update_best(lambda X, w=w, gamma=gamma: predict_mix(X, gamma, w, y_mean, y_std), cv_mse)
        except:
            pass

    # Fallback
    if best_predict is None:
        # Simple linear regression
        try:
            F = np.column_stack([np.ones(n), X_norm])
            A = F.T @ F + 1e-5 * np.eye(F.shape[1])
            w = solve(A, F.T @ y_norm)
            def predict_lin(X, w, y_mean, y_std):
                Xn = (X - X_min) / X_range
                F_test = np.column_stack([np.ones(len(Xn)), Xn])
                return y_mean + y_std * (F_test @ w)
            return lambda X: predict_lin(X, w, y_mean, y_std)
        except:
            return lambda X: np.full(len(X), y_mean)
    
    return best_predict

# EVOLVE-BLOCK-END
