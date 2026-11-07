"""Symbolic regression via multi-basis least-squares with polynomial, trig, exp, and log features."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    t0 = time.time()
    deadline = 330.0
    n, d = X_train.shape
    x = X_train[:, 0].astype(np.float64)
    y = y_train.astype(np.float64)

    # Build a rich set of candidate features
    features_list = []
    feature_names = []

    # Polynomial features up to degree 10
    for p in range(1, 11):
        features_list.append(x ** p)
        feature_names.append(f"x^{p}")

    # Constant
    features_list.append(np.ones_like(x))
    feature_names.append("1")

    # Trigonometric features
    for k in range(1, 5):
        features_list.append(np.sin(k * x))
        features_list.append(np.cos(k * x))
        feature_names.append(f"sin({k}x)")
        feature_names.append(f"cos({k}x)")

    # Exponential features (careful with overflow)
    x_safe = np.clip(x, -50, 50)
    for a in [0.1, 0.2, 0.5, 1.0, 2.0]:
        features_list.append(np.exp(a * x_safe))
        feature_names.append(f"exp({a}x)")
    for a in [0.1, 0.2, 0.5, 1.0, 2.0]:
        features_list.append(np.exp(-a * x_safe))
        feature_names.append(f"exp(-{a}x)")

    # Log features (for positive x, clip to avoid issues)
    x_pos = np.clip(x, 1e-10, None)
    features_list.append(np.log(x_pos))
    feature_names.append("log(x)")
    features_list.append(np.log(x_pos) ** 2)
    feature_names.append("log(x)^2")

    # Sigmoid-like
    features_list.append(1.0 / (1.0 + np.exp(-x_safe)))
    feature_names.append("sigmoid(x)")
    features_list.append(1.0 / (1.0 + np.exp(-2.0 * x_safe)))
    feature_names.append("sigmoid(2x)")

    # Tanh
    features_list.append(np.tanh(x_safe))
    feature_names.append("tanh(x)")
    features_list.append(np.tanh(2.0 * x_safe))
    feature_names.append("tanh(2x)")

    # Absolute value and sign-based
    features_list.append(np.abs(x))
    feature_names.append("|x|")
    features_list.append(np.sign(x) * np.abs(x) ** 0.5)
    feature_names.append("sign(x)*sqrt(|x|)")

    # Reciprocal
    x_rec = np.where(np.abs(x) > 1e-10, 1.0 / x, 0.0)
    features_list.append(x_rec)
    feature_names.append("1/x")
    features_list.append(x_rec ** 2)
    feature_names.append("1/x^2")

    # Square root
    features_list.append(np.sqrt(np.abs(x)))
    feature_names.append("sqrt(|x|)")

    # Combine all features
    A = np.column_stack(features_list)
    n_features = A.shape[1]

    # Remove constant column if it's linearly dependent (it's fine, lstsq handles it)
    # But let's center y for numerical stability in some cases
    y_mean = np.mean(y)
    y_centered = y - y_mean

    # Solve least squares with regularization for stability
    # Use SVD-based solution with rcond
    try:
        # Try unregularized first
        w, residuals, rank, sv = np.linalg.lstsq(A, y_centered, rcond=1e-12)
    except np.linalg.LinAlgError:
        # Fallback to ridge regression
        w = np.zeros(n_features)
        w[-1] = 0  # constant term handled separately

    # Check if the fit is good; if not, try ridge regularization
    pred_centered = A @ w
    mse = np.mean((pred_centered - y_centered) ** 2)
    var_y = np.var(y)
    nrmse = mse / var_y if var_y > 1e-15 else 0.0

    # If the fit is poor, try ridge regression with various lambda values
    best_w = w
    best_nrmse = nrmse
    if nrmse > 0.01:
        for lam in [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]:
            try:
                A_t_A = A.T @ A + lam * np.eye(n_features)
                w_ridge = np.linalg.solve(A_t_A, A.T @ y_centered)
                pred_ridge = A @ w_ridge
                mse_ridge = np.mean((pred_ridge - y_centered) ** 2)
                nrmse_ridge = mse_ridge / var_y if var_y > 1e-15 else 0.0
                if nrmse_ridge < best_nrmse:
                    best_nrmse = nrmse_ridge
                    best_w = w_ridge
            except np.linalg.LinAlgError:
                pass

    w_final = best_w

    def predict(X):
        x_pred = X[:, 0].astype(np.float64)
        # Rebuild features in the same way
        feats = []
        for p in range(1, 11):
            feats.append(x_pred ** p)
        feats.append(np.ones_like(x_pred))
        for k in range(1, 5):
            feats.append(np.sin(k * x_pred))
            feats.append(np.cos(k * x_pred))
        x_safe = np.clip(x_pred, -50, 50)
        for a in [0.1, 0.2, 0.5, 1.0, 2.0]:
            feats.append(np.exp(a * x_safe))
        for a in [0.1, 0.2, 0.5, 1.0, 2.0]:
            feats.append(np.exp(-a * x_safe))
        x_pos = np.clip(x_pred, 1e-10, None)
        feats.append(np.log(x_pos))
        feats.append(np.log(x_pos) ** 2)
        feats.append(1.0 / (1.0 + np.exp(-x_safe)))
        feats.append(1.0 / (1.0 + np.exp(-2.0 * x_safe)))
        feats.append(np.tanh(x_safe))
        feats.append(np.tanh(2.0 * x_safe))
        feats.append(np.abs(x_pred))
        feats.append(np.sign(x_pred) * np.abs(x_pred) ** 0.5)
        x_rec = np.where(np.abs(x_pred) > 1e-10, 1.0 / x_pred, 0.0)
        feats.append(x_rec)
        feats.append(x_rec ** 2)
        feats.append(np.sqrt(np.abs(x_pred)))
        A_pred = np.column_stack(feats)
        return A_pred @ w_final + y_mean

    return predict
# EVOLVE-BLOCK-END
