# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 300s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).flatten()
    n, d = X.shape

    # Normalize features for better kernel conditioning
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-12] = 1.0
    X_norm = (X - X_mean) / X_std

    def _rbf_kernel(A, B, gamma):
        # Efficient RBF: ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
        A_sq = np.sum(A * A, axis=1)[:, None]
        B_sq = np.sum(B * B, axis=1)[None, :]
        dist_sq = A_sq + B_sq - 2.0 * A @ B.T
        np.maximum(dist_sq, 0.0, out=dist_sq)
        return np.exp(-gamma * dist_sq)

    def _poly_kernel(A, B, degree, gamma, coef0):
        return (gamma * A @ B.T + coef0) ** degree

    def _linear_kernel(A, B):
        return A @ B.T

    def _fit_kernel(K_train, alpha):
        # Solve (K + alpha*I) w = y
        n_k = K_train.shape[0]
        A = K_train + alpha * np.eye(n_k)
        try:
            w = np.linalg.solve(A, y)
        except np.linalg.LinAlgError:
            w, *_ = np.linalg.lstsq(A, y, rcond=None)
        return w

    def _make_predictor(K_train, w, kernel_func):
        def predict(X_test):
            X_t = np.asarray(X_test, dtype=np.float64)
            X_t_norm = (X_t - X_mean) / X_std
            K_test = kernel_func(X_t_norm, X_norm)
            return K_test @ w
        return predict

    # Candidate kernels with hyperparameters
    candidates = []

    # RBF kernels with different gammas
    for gamma in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        K = _rbf_kernel(X_norm, X_norm, gamma)
        candidates.append(('rbf', gamma, K))

    # Polynomial kernels
    for degree in [2, 3, 4]:
        for gamma in [0.5, 1.0, 2.0]:
            for coef0 in [1.0, 2.0]:
                K = _poly_kernel(X_norm, X_norm, degree, gamma, coef0)
                candidates.append(('poly', (degree, gamma, coef0), K))

    # Linear kernel
    K_lin = _linear_kernel(X_norm, X_norm)
    candidates.append(('linear', None, K_lin))

    # Fit each candidate with multiple regularization strengths
    best_models = []
    alphas = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]

    for name, params, K in candidates:
        for alpha in alphas:
            if time.time() > deadline - 30.0:
                break
            w = _fit_kernel(K, alpha)
            # Training MSE
            pred_train = K @ w
            mse = np.mean((pred_train - y) ** 2)
            best_models.append((mse, name, params, w, K))

    if not best_models:
        # Fallback: simple linear regression
        w, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), X_norm]), y, rcond=None)
        def predict_fallback(X_test):
            X_t = np.asarray(X_test, dtype=np.float64)
            X_t_norm = (X_t - X_mean) / X_std
            return np.column_stack([np.ones(len(X_t)), X_t_norm]) @ w
        return predict_fallback

    # Select top models and ensemble them
    # Use a simple validation-free ensemble: average predictions from top-k models
    # Weight by inverse MSE (but cap to avoid domination)
    best_models.sort(key=lambda x: x[0])
    top_k = min(5, len(best_models))
    top_models = best_models[:top_k]

    # Compute weights based on inverse MSE with smoothing
    mses = np.array([m[0] for m in top_models])
    # Avoid division by zero
    mses_safe = np.maximum(mses, 1e-12)
    weights = 1.0 / mses_safe
    weights /= weights.sum()

    # Build ensemble predictor
    model_infos = []
    for i, (mse, name, params, w, K) in enumerate(top_models):
        if name == 'rbf':
            gamma = params
            kf = lambda A, B, g=gamma: _rbf_kernel(A, B, g)
        elif name == 'poly':
            degree, gamma, coef0 = params
            kf = lambda A, B, d=degree, g=gamma, c=coef0: _poly_kernel(A, B, d, g, c)
        elif name == 'linear':
            kf = lambda A, B: _linear_kernel(A, B)
        else:
            continue
        model_infos.append((weights[i], kf, w))

    if not model_infos:
        # Fallback to best single model
        _, name, params, w, K = best_models[0]
        if name == 'rbf':
            gamma = params
            kf = lambda A, B, g=gamma: _rbf_kernel(A, B, g)
        elif name == 'poly':
            degree, gamma, coef0 = params
            kf = lambda A, B, d=degree, g=gamma, c=coef0: _poly_kernel(A, B, d, g, c)
        else:
            kf = lambda A, B: _linear_kernel(A, B)
        def predict_single(X_test):
            X_t = np.asarray(X_test, dtype=np.float64)
            X_t_norm = (X_t - X_mean) / X_std
            K_test = kf(X_t_norm, X_norm)
            return K_test @ w
        return predict_single

    def predict(X_test):
        X_t = np.asarray(X_test, dtype=np.float64)
        X_t_norm = (X_t - X_mean) / X_std
        result = np.zeros(len(X_t), dtype=np.float64)
        for wt, kf, w in model_infos:
            K_test = kf(X_t_norm, X_norm)
            result += wt * (K_test @ w)
        return result

    return predict
# EVOLVE-BLOCK-END
