"""Symbolic regression via ensemble of sparse kernel models with adaptive grid search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    # 330s wall-clock limit. Reserve 20s for overhead, leaving 310s.
    deadline = start_time + 310.0

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).flatten()
    n_samples, n_features = X.shape

    # Normalize inputs to [0, 1]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_norm = (X - X_min) / X_range

    # Standardize y
    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    def rbf_kernel(X1, X2, length_scale):
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
        sq_dists = np.maximum(sq_dists, 0)
        return np.exp(-sq_dists / (2 * length_scale**2))

    def poly_kernel(X1, X2, degree, gamma, coef0):
        return (gamma * (X1 @ X2.T) + coef0) ** degree

    def linear_kernel(X1, X2):
        return X1 @ X2.T

    # Candidate hyperparameters
    length_scales = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    degrees = [2, 3, 4]
    gammas = [0.5, 1.0, 2.0, 5.0]
    coef0s = [0.0, 0.5, 1.0]
    alphas = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

    best_train_mse = np.inf
    best_model = None  # (kernel_type, params, alpha_vec)

    # Precompute for speed
    X_norm_sq = np.sum(X_norm**2, axis=1)

    # Grid search over kernel types and hyperparameters
    # To keep time bounded, we limit total number of combinations tried
    max_combos = 200
    combo_count = 0

    for kernel_type in ['rbf', 'poly', 'linear']:
        if time.time() > deadline - 10:
            break

        if kernel_type == 'rbf':
            for ls in length_scales:
                if time.time() > deadline - 10 or combo_count >= max_combos:
                    break
                for alpha in alphas:
                    if time.time() > deadline - 10 or combo_count >= max_combos:
                        break
                    combo_count += 1
                    try:
                        K = rbf_kernel(X_norm, X_norm, ls)
                        K_reg = K + alpha * np.eye(n_samples)
                        L = np.linalg.cholesky(K_reg)
                        y_temp = np.linalg.solve(L, y_norm)
                        alpha_vec = np.linalg.solve(L.T, y_temp)
                        pred_train = K @ alpha_vec
                        mse = np.mean((pred_train - y_norm)**2)
                        if mse < best_train_mse:
                            best_train_mse = mse
                            best_model = ('rbf', (ls,), alpha_vec)
                    except Exception:
                        pass

        elif kernel_type == 'poly':
            for degree in degrees:
                if time.time() > deadline - 10 or combo_count >= max_combos:
                    break
                for gamma in gammas:
                    if time.time() > deadline - 10 or combo_count >= max_combos:
                        break
                    for coef0 in coef0s:
                        if time.time() > deadline - 10 or combo_count >= max_combos:
                            break
                        for alpha in alphas[:2]:  # Fewer alphas for poly to save time
                            if time.time() > deadline - 10 or combo_count >= max_combos:
                                break
                            combo_count += 1
                            try:
                                K = poly_kernel(X_norm, X_norm, degree, gamma, coef0)
                                K_reg = K + alpha * np.eye(n_samples)
                                L = np.linalg.cholesky(K_reg)
                                y_temp = np.linalg.solve(L, y_norm)
                                alpha_vec = np.linalg.solve(L.T, y_temp)
                                pred_train = K @ alpha_vec
                                mse = np.mean((pred_train - y_norm)**2)
                                if mse < best_train_mse:
                                    best_train_mse = mse
                                    best_model = ('poly', (degree, gamma, coef0), alpha_vec)
                            except Exception:
                                pass

        elif kernel_type == 'linear':
            for alpha in alphas:
                if time.time() > deadline - 10 or combo_count >= max_combos:
                    break
                combo_count += 1
                try:
                    K = linear_kernel(X_norm, X_norm)
                    K_reg = K + alpha * np.eye(n_samples)
                    L = np.linalg.cholesky(K_reg)
                    y_temp = np.linalg.solve(L, y_norm)
                    alpha_vec = np.linalg.solve(L.T, y_temp)
                    pred_train = K @ alpha_vec
                    mse = np.mean((pred_train - y_norm)**2)
                    if mse < best_train_mse:
                        best_train_mse = mse
                        best_model = ('linear', (), alpha_vec)
                except Exception:
                    pass

    # Fallback if no model found
    if best_model is None:
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            return np.full(X_test.shape[0], y_mean)
        return predict

    kernel_type, kernel_params, alpha_vec = best_model

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        X_test_norm = (X_test - X_min) / X_range

        if kernel_type == 'rbf':
            ls = kernel_params[0]
            sq_dists = np.sum(X_test_norm**2, axis=1)[:, None] + np.sum(X_norm**2, axis=1)[None, :] - 2 * X_test_norm @ X_norm.T
            sq_dists = np.maximum(sq_dists, 0)
            K_tt = np.exp(-sq_dists / (2 * ls**2))
        elif kernel_type == 'poly':
            degree, gamma, coef0 = kernel_params
            K_tt = (gamma * (X_test_norm @ X_norm.T) + coef0) ** degree
        elif kernel_type == 'linear':
            K_tt = X_test_norm @ X_norm.T

        pred_norm = K_tt @ alpha_vec
        pred = pred_norm * y_std + y_mean
        return pred

    return predict
# EVOLVE-BLOCK-END
