"""Climate dynamics: unresolved-tendency closure with Tikhonov-regularized polynomial fit."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing.
    Uses Tikhonov regularization (L2 penalty on coefficients, excluding intercept)
    with the regularization strength selected via internal k-fold cross-validation.
    """
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s wall-clock limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    n = len(x)

    if n < 6:
        # Not enough data for degree-5; fall back to linear
        try:
            c = np.polynomial.polynomial.polyfit(x, y, 1)
            out = np.zeros(6)
            out[:2] = c
            return out
        except Exception:
            return np.zeros(6)

    # Build the Vandermonde matrix for degree 5 (ascending powers)
    # V[:, i] = x^i
    V = np.vander(x, N=6, increasing=True)  # shape (n, 6)

    # Regularization matrix: penalty on coefficients 1..5 (exclude intercept)
    # Lambda identity but with 0 in the (0,0) position
    reg_mask = np.zeros((6, 6))
    reg_mask[1:, 1:] = np.eye(5)

    def solve_regularized(V_k, y_k, lam):
        """Solve (V^T V + lam * R) c = V^T y where R is the regularization matrix."""
        n_k = V_k.shape[0]
        A = V_k.T @ V_k  # (6,6)
        b = V_k.T @ y_k  # (6,)
        A_reg = A + lam * reg_mask
        try:
            c = np.linalg.solve(A_reg, b)
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(A_reg, b, rcond=None)[0]
        return c

    # Cross-validation to select lambda
    # Use 5-fold CV
    n_folds = 5
    fold_size = n // n_folds
    indices = np.arange(n)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)

    # Log-spaced lambda grid
    lam_grid = np.logspace(-8, 2, 30)  # 30 values from 1e-8 to 1e2

    best_lam = 0.0
    best_cv_mse = np.inf

    for lam in lam_grid:
        if time.time() > deadline - 20:
            break
        cv_mses = []
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

            V_train = V[train_idx]
            y_train = y[train_idx]
            V_val = V[val_idx]
            y_val = y[val_idx]

            c = solve_regularized(V_train, y_train, lam)
            pred = V_val @ c
            mse = np.mean((pred - y_val) ** 2)
            cv_mses.append(mse)

        mean_cv_mse = np.mean(cv_mses)
        if mean_cv_mse < best_cv_mse:
            best_cv_mse = mean_cv_mse
            best_lam = lam

    # Final fit on all data with best lambda
    c = solve_regularized(V, y, best_lam)

    # Ensure finite
    if not np.all(np.isfinite(c)):
        c = np.zeros(6)
        try:
            c[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except Exception:
            pass

    return c
# EVOLVE-BLOCK-END
