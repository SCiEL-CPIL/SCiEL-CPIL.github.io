"""Symbolic regression over 10 hidden equations using an ensemble of kernel methods."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial.distance import cdist

_DEADLINE = time.time() + 320  # safety margin before 330s kill


def _rbf_kernel(X, Y, gamma):
    """Compute RBF kernel matrix between X and Y."""
    d = cdist(X, Y, 'sqeuclidean')
    return np.exp(-gamma * d)


def _fit_ridge(K, y, alpha):
    """Solve ridge regression: (K + alpha*I) w = y."""
    n = K.shape[0]
    A = K + alpha * np.eye(n)
    try:
        w = np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, y, rcond=None)[0]
    return w


def _poly_features(X, degree=3):
    """Generate polynomial features up to given degree."""
    n, d = X.shape
    cols = [np.ones((n, 1))]
    for deg in range(1, degree + 1):
        for i in range(d):
            for j in range(i, d):
                if deg == 1:
                    if i == j:
                        cols.append(X[:, i:i+1])
                elif deg == 2:
                    cols.append((X[:, i] * X[:, j]).reshape(-1, 1))
                elif deg == 3:
                    if i == j:
                        cols.append((X[:, i]**3).reshape(-1, 1))
                    else:
                        cols.append((X[:, i]**2 * X[:, j]).reshape(-1, 1))
                        cols.append((X[:, i] * X[:, j]**2).reshape(-1, 1))
    return np.column_stack(cols)


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n, d = X_train.shape
    t0 = time.time()

    # Standardize inputs for better kernel scaling
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std < 1e-12] = 1.0
    X_train_std = (X_train - X_mean) / X_std

    # Fit multiple RBF models with different gammas and ridge alphas
    gammas = [0.5, 1.0, 2.0, 5.0, 10.0]
    alphas = [0.01, 0.1, 1.0]

    models = []  # list of (K_train, w, gamma)
    for gamma in gammas:
        if time.time() - t0 > _DEADLINE - 10:
            break
        K_train = _rbf_kernel(X_train_std, X_train_std, gamma)
        for alpha in alphas:
            if time.time() - t0 > _DEADLINE - 5:
                break
            w = _fit_ridge(K_train, y_train, alpha)
            models.append((K_train, w, gamma))

    # Also fit a polynomial ridge model as fallback
    poly_w = None
    if time.time() - t0 < _DEADLINE - 5:
        try:
            P_train = _poly_features(X_train_std, degree=3)
            alpha_p = 0.1
            A_p = P_train.T @ P_train + alpha_p * np.eye(P_train.shape[1])
            poly_w = np.linalg.solve(A_p, P_train.T @ y_train)
        except Exception:
            poly_w = None

    # Select best model via cross-validation on training data
    best_model_idx = 0
    best_cv_mse = np.inf
    cv_folds = 5
    n_cv = max(1, n // cv_folds)

    if models:
        for idx, (K_tr, w, gamma) in enumerate(models):
            if time.time() - t0 > _DEADLINE - 5:
                break
            cv_errors = []
            for fold in range(cv_folds):
                val_idx = np.arange(fold * n_cv, (fold + 1) * n_cv)
                tr_idx = np.setdiff1d(np.arange(n), val_idx)
                # K_val_train: kernel between val and tr
                X_val_std = X_train_std[val_idx]
                X_tr_std = X_train_std[tr_idx]
                K_val_tr = _rbf_kernel(X_val_std, X_tr_std, gamma)
                y_tr = y_train[tr_idx]
                # Re-fit on subset quickly using precomputed K
                K_sub = K_tr[np.ix_(tr_idx, tr_idx)]
                alpha = 0.1  # use middle alpha for CV
                w_sub = _fit_ridge(K_sub, y_tr, alpha)
                pred_val = K_val_tr @ w_sub
                mse = np.mean((pred_val - y_train[val_idx])**2)
                cv_errors.append(mse)
            avg_mse = np.mean(cv_errors)
            if avg_mse < best_cv_mse:
                best_cv_mse = avg_mse
                best_model_idx = idx

    # Also evaluate polynomial model via CV
    if poly_w is not None and time.time() - t0 < _DEADLINE - 5:
        cv_errors = []
        for fold in range(cv_folds):
            val_idx = np.arange(fold * n_cv, (fold + 1) * n_cv)
            tr_idx = np.setdiff1d(np.arange(n), val_idx)
            P_tr = _poly_features(X_train_std[tr_idx], degree=3)
            P_val = _poly_features(X_train_std[val_idx], degree=3)
            alpha_p = 0.1
            A_p = P_tr.T @ P_tr + alpha_p * np.eye(P_tr.shape[1])
            w_sub = np.linalg.solve(A_p, P_tr.T @ y_train[tr_idx])
            pred_val = P_val @ w_sub
            mse = np.mean((pred_val - y_train[val_idx])**2)
            cv_errors.append(mse)
        avg_mse = np.mean(cv_errors)
        if avg_mse < best_cv_mse:
            best_cv_mse = avg_mse
            best_model_idx = -1  # special marker for polynomial

    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        X_std = (X - X_mean) / X_std
        if best_model_idx == -1:
            # Use polynomial model
            P = _poly_features(X_std, degree=3)
            return P @ poly_w
        else:
            K_train, w, gamma = models[best_model_idx]
            K_test = _rbf_kernel(X_std, X_train_std, gamma)
            return K_test @ w

    return predict
# EVOLVE-BLOCK-END
