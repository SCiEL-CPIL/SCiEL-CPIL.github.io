"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    t_start = time.time()
    deadline = 295.0  # safety margin before 300s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Normalize inputs
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    # Build a rich finite feature basis combining polynomial, trig, exp, log
    def build_features(X_in, max_poly_deg=4, n_freq=5):
        """Build a rich feature matrix from input X_in (n, d)."""
        n_rows = X_in.shape[0]
        cols = [np.ones(n_rows)]

        # Polynomial features up to max_poly_deg
        from itertools import combinations_with_replacement
        for degree in range(1, max_poly_deg + 1):
            for combo in combinations_with_replacement(range(d), degree):
                term = np.ones(n_rows)
                for idx in combo:
                    term = term * X_in[:, idx]
                cols.append(term)

        # Per-dimension nonlinear transforms
        for j in range(d):
            xj = X_in[:, j]
            xj_safe = np.clip(xj, 1e-10, None)
            cols.append(np.sin(xj))
            cols.append(np.cos(xj))
            cols.append(np.log(xj_safe))
            cols.append(np.sqrt(xj_safe))
            cols.append(1.0 / xj_safe)
            cols.append(xj * np.log(xj_safe))
            cols.append(np.exp(-xj / 5.0))
            cols.append(np.exp(-xj / 10.0))

        # Interaction terms between different dimensions (degree 2 only, cross)
        if d >= 2:
            for i in range(d):
                for j in range(i + 1, d):
                    cols.append(X_in[:, i] * X_in[:, j])

        F = np.column_stack(cols)
        return F

    def ridge_fit(X_feat, y_vec, alpha):
        """Ridge regression solve."""
        A = X_feat.T @ X_feat + alpha * np.eye(X_feat.shape[1])
        b = X_feat.T @ y_vec
        try:
            w = np.linalg.solve(A, b)
        except:
            w = np.linalg.lstsq(X_feat, y_vec, rcond=None)[0]
        return w

    def cross_validate(X_feat, y_vec, alpha, n_folds=5, seed=42):
        """K-fold CV on precomputed features."""
        rng = np.random.RandomState(seed)
        indices = np.arange(X_feat.shape[0])
        rng.shuffle(indices)
        fold_size = X_feat.shape[0] // n_folds
        errors = []
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else X_feat.shape[0]
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

            X_tr = X_feat[train_idx]
            y_tr = y_vec[train_idx]
            X_va = X_feat[val_idx]
            y_va = y_vec[val_idx]

            w = ridge_fit(X_tr, y_tr, alpha)
            pred = X_va @ w
            mse = np.mean((pred - y_va) ** 2)
            errors.append(mse)
        return np.mean(errors)

    # Try multiple feature configurations and regularization strengths
    alpha_options = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    poly_deg_options = [2, 3, 4]

    best_cv = np.inf
    best_predictor = None
    best_config = None

    for pdeg in poly_deg_options:
        if time.time() > deadline - 20:
            break
        try:
            F = build_features(X_norm, max_poly_deg=pdeg)
        except:
            continue

        n_feat = F.shape[1]
        if n_feat > 2000:
            continue

        for alpha in alpha_options:
            if time.time() > deadline - 10:
                break
            try:
                cv_score = cross_validate(F, y, alpha)
                if cv_score < best_cv:
                    best_cv = cv_score
                    best_config = (pdeg, alpha)
                    # Fit on full data
                    w_full = ridge_fit(F, y, alpha)
                    F_ref = F.copy()
                    X_mean_ref = X_mean.copy()
                    X_std_ref = X_std.copy()
                    pdeg_ref = pdeg

                    def predict(X_pred, _F_ref=F_ref, _X_mean=X_mean_ref,
                                _X_std=X_std_ref, _pdeg=pdeg_ref, _w=w_full):
                        Xp = np.asarray(X_pred, dtype=np.float64)
                        Xp_norm = (Xp - _X_mean) / _X_std
                        Fp = build_features(Xp_norm, max_poly_deg=_pdeg)
                        return Fp @ _w

                    best_predictor = predict
            except:
                continue

    if best_predictor is not None:
        return best_predictor

    # Fallback: simple linear regression
    F_lin = np.column_stack([np.ones(n), X_norm])
    w_lin = ridge_fit(F_lin, y, 1e-4)

    def predict(X_pred):
        Xp = np.asarray(X_pred, dtype=np.float64)
        Xp_norm = (Xp - X_mean) / X_std
        Fp = np.column_stack([np.ones(Xp_norm.shape[0]), Xp_norm])
        return Fp @ w_lin

    return predict
# EVOLVE-BLOCK-END
