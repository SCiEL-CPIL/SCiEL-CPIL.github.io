# EVOLVE-BLOCK-START
import numpy as np
import time


def _rbf_kernel(X1, X2, length_scale):
    """Compute RBF kernel between two sets of points."""
    sq1 = np.sum(X1 ** 2, axis=1)[:, None]
    sq2 = np.sum(X2 ** 2, axis=1)[None, :]
    d2 = sq1 + sq2 - 2.0 * (X1 @ X2.T)
    d2 = np.maximum(d2, 0.0)
    return np.exp(-d2 / (2.0 * length_scale ** 2))


def _poly_features(X, degree=3):
    """Create polynomial features up to given degree."""
    d = X.shape[1]
    cols = [np.ones(len(X))]
    # degree 1
    for j in range(d):
        cols.append(X[:, j])
    # degree 2
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    # degree 3
    if degree >= 3:
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    cols.append(X[:, i] * X[:, j] * X[:, k])
    return np.column_stack(cols)


def _poly_ridge(X_train, y_train, X_test, alpha=1e-3):
    """Ridge regression on polynomial features."""
    F_train = _poly_features(X_train, degree=3)
    F_test = _poly_features(X_test, degree=3)
    # Ridge: solve (F^T F + alpha I) w = F^T y
    n_feat = F_train.shape[1]
    A = F_train.T @ F_train + alpha * np.eye(n_feat)
    b = F_train.T @ y_train
    w = np.linalg.solve(A, b)
    return F_test @ w


def _rbf_ridge(X_train, y_train, X_test, length_scale, alpha=1e-3):
    """Kernel ridge regression with RBF kernel."""
    K_train = _rbf_kernel(X_train, X_train, length_scale)
    K_test = _rbf_kernel(X_test, X_train, length_scale)
    n = len(X_train)
    A = K_train + alpha * np.eye(n)
    b = y_train
    w = np.linalg.solve(A, b)
    return K_test @ w


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n, d = X_train.shape
    t0 = time.time()
    deadline = t0 + 280.0  # safety margin before 300s limit

    # Normalize features for better kernel behavior
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-12] = 1.0
    X_train_norm = (X_train - mean) / std

    # Compute baseline: polynomial ridge
    # We need a placeholder; we'll store predictions per candidate model
    # Strategy: try several length scales for RBF, pick best by train fit (noiseless),
    # but blend with polynomial to ensure robustness.

    # Candidate length scales (on normalized space)
    length_scales = [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0]

    best_pred_train = None
    best_score = np.inf
    best_length = None

    # Evaluate RBF ridge candidates on training data (noiseless, so train fit is a proxy)
    for ls in length_scales:
        if time.time() > deadline:
            break
        try:
            K_train = _rbf_kernel(X_train_norm, X_train_norm, ls)
            alpha = 1e-3
            A = K_train + alpha * np.eye(n)
            w = np.linalg.solve(A, y_train)
            pred_train = K_train @ w
            # Noiseless: residual should be ~0 for good fit. Use residual as proxy.
            resid = np.mean((pred_train - y_train) ** 2)
            if resid < best_score:
                best_score = resid
                best_length = ls
                best_pred_train = pred_train
        except Exception:
            continue

    # Also compute polynomial ridge train residual
    poly_w = None
    try:
        F_train = _poly_features(X_train, degree=3)
        alpha = 1e-3
        n_feat = F_train.shape[1]
        A = F_train.T @ F_train + alpha * np.eye(n_feat)
        b = F_train.T @ y_train
        poly_w = np.linalg.solve(A, b)
        poly_pred_train = F_train @ poly_w
        poly_resid = np.mean((poly_pred_train - y_train) ** 2)
    except Exception:
        poly_resid = np.inf
        poly_pred_train = None

    # Decide: if RBF train residual is good (close to 0), use RBF; else blend/fallback to poly.
    # Since noiseless, a good model should have near-zero train residual.
    use_rbf = (best_length is not None) and (best_score < 1e-6)
    use_poly = poly_resid < 1e-6

    # We'll create a predict function that blends: if RBF fits well, use it; else poly; else mean.
    # To be safe, blend RBF and poly with weights based on train residual.
    # Actually, since noiseless, pick the one with lower train residual. If both bad, use poly (more stable).

    if use_rbf and (not use_poly or best_score <= poly_resid):
        chosen = 'rbf'
        chosen_ls = best_length
    elif use_poly:
        chosen = 'poly'
    elif use_rbf:
        chosen = 'rbf'
        chosen_ls = best_length
    else:
        # Fallback: use polynomial ridge (more stable than RBF with bad length)
        chosen = 'poly'

    # Precompute for speed
    if chosen == 'rbf':
        # Precompute K_train and w
        K_train = _rbf_kernel(X_train_norm, X_train_norm, chosen_ls)
        alpha = 1e-3
        A = K_train + alpha * np.eye(n)
        w = np.linalg.solve(A, y_train)

        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            X_norm = (X - mean) / std
            K_test = _rbf_kernel(X_norm, X_train_norm, chosen_ls)
            return K_test @ w
    else:
        # Polynomial ridge
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            F_test = _poly_features(X, degree=3)
            return F_test @ poly_w

    return predict
# EVOLVE-BLOCK-END
