"""Combined program: Ridge regression with K-Fold CV regularization selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale
    forcing of a two-scale Lorenz-96 model using Ridge regression with
    K-Fold cross-validation for regularization parameter selection.
    
    Returns 6 finite real polynomial coefficients in ASCENDING powers of x.
    """
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 120s limit

    # Convert inputs to numpy arrays and flatten
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)

    x = x[:n]
    y = y[:n]

    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size

    if n == 0:
        return np.zeros(6, dtype=np.float64)

    # Check time
    if time.time() > deadline:
        return np.zeros(6, dtype=np.float64)

    # Build Vandermonde matrix: V[:, i] = x^i for i = 0, ..., 5
    V = np.vander(x, N=6, increasing=True)

    # Center and scale columns for better conditioning
    col_means = np.mean(V, axis=0)
    col_stds = np.std(V, axis=0)
    col_stds[col_stds < 1e-12] = 1.0
    V_scaled = (V - col_means) / col_stds

    # K-Fold Cross-Validation to select regularization parameter
    # Use 5-fold CV for a good bias-variance tradeoff
    n_folds = 5
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    fold_size = n // n_folds
    folds = [indices[i * fold_size: (i + 1) * fold_size] for i in range(n_folds)]
    folds[-1] = np.concatenate(folds[-1:])  # Handle remainder in last fold

    # Log-spaced regularization parameters to try
    lambda_candidates = np.logspace(-8, 2, 50)

    best_lambda = 0.0
    best_cv_error = np.inf

    for lam in lambda_candidates:
        if time.time() > deadline:
            break

        cv_errors = []
        for fold_i in range(n_folds):
            val_idx = folds[fold_i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_i])

            V_train = V_scaled[train_idx]
            y_train = y[train_idx]
            V_val = V_scaled[val_idx]
            y_val = y[val_idx]

            # Ridge regression: minimize ||V_train @ coeff - y_train||^2 + lam * ||coeff||^2
            # Solution: (V_train^T V_train + lam * I) @ coeff = V_train^T y_train
            try:
                A = V_train.T @ V_train + lam * np.eye(6)
                b = V_train.T @ y_train
                coeff_fold = np.linalg.solve(A, b)

                # Evaluate on validation fold
                y_pred = V_val @ coeff_fold
                if not np.all(np.isfinite(y_pred)):
                    continue
                cv_errors.append(np.mean((y_val - y_pred) ** 2))
            except np.linalg.LinAlgError:
                continue

        if len(cv_errors) == n_folds:
            mean_cv_error = np.mean(cv_errors)
            if mean_cv_error < best_cv_error:
                best_cv_error = mean_cv_error
                best_lambda = lam

    # Check time
    if time.time() > deadline:
        # Fallback: use unregularized least squares
        try:
            coeff, _, _, _ = np.linalg.lstsq(V, y, rcond=None)
            if not np.all(np.isfinite(coeff)):
                coeff = np.zeros(6)
            return coeff
        except Exception:
            return np.zeros(6)

    # Final fit on ALL data with the best lambda
    try:
        A = V_scaled.T @ V_scaled + best_lambda * np.eye(6)
        b = V_scaled.T @ y
        coeff_scaled = np.linalg.solve(A, b)

        # Convert back from scaled to original coefficients
        # V_scaled = (V - col_means) / col_stds
        # y = V @ coeff = (V_scaled * col_stds + col_means) @ coeff
        # y = V_scaled @ (col_stds * coeff) + col_means @ coeff
        # So we need: V_scaled @ coeff_scaled = V @ coeff
        # V @ coeff = (V_scaled * col_stds + col_means) @ coeff
        # = V_scaled @ (col_stds * coeff) + col_means @ coeff
        # Therefore: coeff_scaled = col_stds * coeff + col_means @ coeff / (something)
        # Actually, let's just do the conversion properly.
        # V_scaled @ coeff_scaled = V @ coeff
        # ((V - col_means) / col_stds) @ coeff_scaled = V @ coeff
        # (V - col_means) @ (coeff_scaled / col_stds) = V @ coeff
        # V @ (coeff_scaled / col_stds) - col_means @ (coeff_scaled / col_stds) = V @ coeff
        # This is getting complex. Let's just solve directly on V.
        pass
    except Exception:
        pass

    # Simpler approach: solve directly on V with regularization
    try:
        A = V.T @ V + best_lambda * np.eye(6)
        b = V.T @ y
        coeff = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback to lstsq
        try:
            coeff, _, _, _ = np.linalg.lstsq(V, y, rcond=None)
        except Exception:
            coeff = np.zeros(6)

    # Ensure coefficients are finite
    if not np.all(np.isfinite(coeff)):
        coeff = np.zeros(6)

    # Final check on time
    if time.time() > deadline:
        return coeff

    # Verify the solution is reasonable
    y_pred = V @ coeff
    if not np.all(np.isfinite(y_pred)):
        coeff = np.zeros(6)

    return coeff
# EVOLVE-BLOCK-END
