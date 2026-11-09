"""Rewritten symbolic-regression fitter using kernel ridge regression with composite kernels."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import solve


def fit(X_train, y_train):
    t_start = time.time()
    deadline = 270.0  # safety margin before 300s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Normalize X for better kernel behavior
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X - X_mean) / X_std

    def compute_kernel(X1, X2, length_scale, degree, periodicity):
        """Compute composite kernel: RBF + polynomial + periodic."""
        # RBF kernel
        X1_norm = X1 - X_mean
        X1_norm = X1_norm / X_std
        X2_norm = X2 - X_mean
        X2_norm = X2_norm / X_std
        sq_dists = (
            np.sum(X1_norm ** 2, axis=1)[:, None] +
            np.sum(X2_norm ** 2, axis=1)[None, :] -
            2 * X1_norm @ X2_norm.T
        )
        sq_dists = np.maximum(sq_dists, 0)
        K_rbf = np.exp(-sq_dists / (2 * length_scale ** 2))

        # Polynomial kernel
        poly_base = X1_norm @ X2_norm.T + 1.0
        K_poly = poly_base ** degree

        # Periodic kernel
        if periodicity > 0:
            dist_sq = np.sqrt(np.maximum(sq_dists, 0))
            K_periodic = np.cos(np.pi * dist_sq / periodicity) ** 2
        else:
            K_periodic = np.zeros_like(K_rbf)

        return K_rbf + K_poly + K_periodic

    # Grid search over kernel hyperparameters
    length_scales = [0.5, 1.0, 2.0, 5.0]
    degrees = [2, 3, 4]
    periodicities = [0, 2.0, 5.0]

    best_score = np.inf
    best_params = None
    best_alpha = 1e-6

    # Compute training kernel once for the best params later
    K_train_base = None

    # We'll do a small grid search
    for ls in length_scales:
        for deg in degrees:
            for per in periodicities:
                if time.time() > deadline - 30:
                    break
                K_train = compute_kernel(X_norm, X_norm, ls, deg, per)
                # Add small regularization for numerical stability
                alpha = 1e-6
                K_reg = K_train + alpha * np.eye(n)

                try:
                    # Solve for weights
                    w = solve(K_reg, y, assume_a='pos')
                    # Compute training MSE (proxy for test performance)
                    pred_train = K_train @ w
                    mse_train = np.mean((pred_train - y) ** 2)

                    # Use a simple validation approach: leave-one-out approximation
                    # Or just use training MSE with regularization strength
                    # For speed, we'll use training MSE as proxy
                    if mse_train < best_score:
                        best_score = mse_train
                        best_params = (ls, deg, per)
                        best_alpha = alpha
                        K_train_base = K_train
                except:
                    continue

    # If no valid params found, use defaults
    if best_params is None:
        best_params = (1.0, 3, 0)
        best_alpha = 1e-6
        K_train_base = compute_kernel(X_norm, X_norm, *best_params)

    ls, deg, per = best_params
    alpha = best_alpha

    # Solve for final weights
    K_reg = K_train_base + alpha * np.eye(n)
    try:
        w = solve(K_reg, y, assume_a='pos')
    except:
        # Fallback to ridge regression
        w = np.linalg.lstsq(K_train_base, y, rcond=None)[0]

    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        X_norm = (X - X_mean) / X_std
        K_test = compute_kernel(X_norm, X_norm, *best_params) if X.shape[0] == n else compute_kernel(X_norm, (X_train - X_mean) / X_std, *best_params)
        return K_test @ w

    return predict
# EVOLVE-BLOCK-END
