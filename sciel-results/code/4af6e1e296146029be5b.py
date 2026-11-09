"""Symbolic regression with adaptive GP (RBF) + ridge regression ensemble."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import solve
import warnings


def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = 320.0  # safety margin before 330s kill
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Standardize for numerical stability
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd

    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    # Normalize X to [0,1] for kernel scaling
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    X_norm = (X - X_min) / X_range

    def check_time():
        return time.time() < DEADLINE - 5

    # =====================
    # MODEL 1: Ridge with hybrid basis
    # =====================
    def make_features(Xraw):
        Xr = np.asarray(Xraw, dtype=np.float64)
        Xrn = (Xr - mu) / sd
        cols = [np.ones(len(Xr))]
        # Linear terms
        for j in range(d):
            cols.append(Xrn[:, j])
        # Quadratic terms
        for i in range(d):
            for j in range(i, d):
                cols.append(Xrn[:, i] * Xrn[:, j])
        # Cubic terms
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    cols.append(Xrn[:, i] * Xrn[:, j] * Xrn[:, k])
        # Trig terms
        for j in range(d):
            cols.append(np.sin(Xrn[:, j]))
            cols.append(np.cos(Xrn[:, j]))
        return np.column_stack(cols)

    F = make_features(X)
    best_ridge_score = float('inf')
    best_ridge_w = None

    for log_lam in np.linspace(-6, 6, 13):
        if not check_time():
            break
        lam = np.exp(log_lam)
        try:
            A = F.T @ F + lam * np.eye(F.shape[1])
            w = solve(A, F.T @ y_norm)
            pred = F @ w
            score = np.mean((pred - y_norm) ** 2)
            if score < best_ridge_score:
                best_ridge_score = score
                best_ridge_w = w
        except Exception:
            pass

    # =====================
    # MODEL 2: GP (RBF kernel) with LOO-CV
    # =====================
    def compute_loo_mse_kernel(K, y, lam):
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
        except Exception:
            return np.inf

    def rbf_kernel(X1, X2, gamma):
        sq_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1)[np.newaxis, :] - 2 * (X1 @ X2.T)
        sq_dists = np.maximum(sq_dists, 0)
        return np.exp(-gamma * sq_dists)

    best_gp_score = float('inf')
    best_gp_params = None  # (gamma, lam)

    # Try a range of gamma values
    for log_gamma in np.linspace(-4, 4, 15):
        if not check_time():
            break
        gamma = np.exp(log_gamma)
        K = rbf_kernel(X_norm, X_norm, gamma)
        # Try a few lambda values
        for log_lam in np.linspace(-4, 4, 9):
            if not check_time():
                break
            lam = np.exp(log_lam)
            cv_mse = compute_loo_mse_kernel(K, y_norm, lam)
            if cv_mse < best_gp_score:
                best_gp_score = cv_mse
                best_gp_params = (gamma, lam)

    # =====================
    # MODEL 3: Fourier features with ridge
    # =====================
    def get_fourier_features(X, max_freq):
        cols = [np.ones(len(X))]
        for j in range(X.shape[1]):
            for k in range(1, max_freq + 1):
                cols.append(np.sin(2 * np.pi * k * X[:, j]))
                cols.append(np.cos(2 * np.pi * k * X[:, j]))
        return np.column_stack(cols)

    best_fourier_score = float('inf')
    best_fourier_params = None  # (freq, lam)

    for freq in [2, 3, 4, 5]:
        if not check_time():
            break
        try:
            F4 = get_fourier_features(X_norm, freq)
            for log_lam in np.linspace(-4, 4, 9):
                if not check_time():
                    break
                lam = np.exp(log_lam)
                try:
                    A = F4.T @ F4 + lam * np.eye(F4.shape[1])
                    w = solve(A, F4.T @ y_norm)
                    pred = F4 @ w
                    score = np.mean((pred - y_norm) ** 2)
                    if score < best_fourier_score:
                        best_fourier_score = score
                        best_fourier_params = (freq, lam)
                except Exception:
                    pass
        except Exception:
            pass

    # =====================
    # Select best model
    # =====================
    # Use CV scores for comparison; fall back to train score if needed
    # For ridge, use train score as proxy (no CV computed)
    # For GP, use LOO CV score
    # For Fourier, use train score

    # Convert to comparable scale: all are in normalized y space
    scores = {
        'ridge': best_ridge_score,
        'gp': best_gp_score,
        'fourier': best_fourier_score
    }

    # Pick the model with lowest score
    best_model_name = min(scores, key=scores.get)
    best_score = scores[best_model_name]

    # =====================
    # Build predict function
    # =====================
    def predict(X):
        if time.time() > DEADLINE:
            return np.full(len(X), y_mean)

        Xr = np.asarray(X, dtype=np.float64)
        Xrn = (Xr - mu) / sd
        Xn = (Xr - X_min) / X_range

        if best_model_name == 'ridge':
            F_pred = make_features(Xr)
            pred_norm = F_pred @ best_ridge_w
            return pred_norm * y_std + y_mean

        elif best_model_name == 'gp':
            gamma, lam = best_gp_params
            K_train = rbf_kernel(X_norm, X_norm, gamma)
            K_test = rbf_kernel(Xn, X_norm, gamma)
            try:
                A = K_train + lam * np.eye(n)
                w = solve(A, y_norm)
                pred_norm = K_test @ w
            except Exception:
                return np.full(len(X), y_mean)
            return pred_norm * y_std + y_mean

        elif best_model_name == 'fourier':
            freq, lam = best_fourier_params
            F4 = get_fourier_features(Xn, freq)
            F4_train = get_fourier_features(X_norm, freq)
            try:
                A = F4_train.T @ F4_train + lam * np.eye(F4_train.shape[1])
                w = solve(A, F4_train.T @ y_norm)
                pred_norm = F4 @ w
            except Exception:
                return np.full(len(X), y_mean)
            return pred_norm * y_std + y_mean

        else:
            return np.full(len(X), y_mean)

    return predict
# EVOLVE-BLOCK-END
