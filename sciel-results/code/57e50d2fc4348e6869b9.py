"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved tendency using
    Tikhonov regularization with the regularization strength selected via
    internal k-fold cross-validation.
    """
    start_time = time.time()
    deadline = 120.0  # wall-clock limit in seconds

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Basic sanity: if inputs are degenerate, return zeros
    if x.size == 0 or y.size == 0:
        return np.zeros(6)

    # Ensure 1-D
    x = x.ravel()
    y = y.ravel()

    n = x.size
    if n != y.size:
        # mismatched lengths – truncate to min
        m = min(n, y.size)
        x = x[:m]
        y = y[:m]
        n = m

    if n < 6:
        # Not enough points for degree 5; fall back to linear
        if n < 2:
            return np.zeros(6)
        coeff = np.zeros(6)
        # polyfit expects degree < n
        deg = min(1, n - 1)
        try:
            coeff[:deg+1] = np.polynomial.polynomial.polyfit(x, y, deg)
        except Exception:
            pass
        return coeff

    # Center and scale x for numerical stability of the Vandermonde
    x_mean = x.mean()
    x_std = x.std()
    if x_std < 1e-12:
        x_std = 1.0
    x_scaled = (x - x_mean) / x_std

    # Build Vandermonde matrix for degree 5 in scaled x
    def make_vandermonde(xs):
        # Returns (n, 6) matrix with columns xs**0 .. xs**5
        V = np.empty((xs.size, 6), dtype=np.float64)
        V[:, 0] = 1.0
        for k in range(1, 6):
            V[:, k] = V[:, k-1] * xs
        return V

    V = make_vandermonde(x_scaled)

    # Candidate regularization strengths (log-spaced)
    # lambda is the Tikhonov penalty: min ||V c - y||^2 + lambda * ||c||^2
    # We'll search over a grid of lambda values.
    lambda_grid = np.logspace(-10, 4, 40)  # 40 candidates

    # K-fold cross-validation to pick lambda
    k = 5
    n_folds = n
    fold_size = n_folds // k
    indices = np.arange(n_folds)

    # Shuffle deterministically
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_folds)

    best_lambda = 0.0
    best_cv_mse = np.inf

    for lam in lambda_grid:
        if time.time() > deadline - 10.0:
            break
        cv_mses = []
        for fold in range(k):
            # Define validation indices for this fold
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < k - 1 else n_folds
            val_idx = perm[val_start:val_end]
            train_idx = np.concatenate([perm[:val_start], perm[val_end:]])

            V_train = V[train_idx]
            y_train = y[train_idx]
            V_val = V[val_idx]
            y_val = y[val_idx]

            # Solve regularized least squares: (V^T V + lambda*I) c = V^T y
            # Use the normal equations with ridge
            try:
                A = V_train.T @ V_train + lam * np.eye(6)
                b = V_train.T @ y_train
                c = np.linalg.solve(A, b)
                pred_val = V_val @ c
                mse = np.mean((pred_val - y_val) ** 2)
            except np.linalg.LinAlgError:
                mse = np.inf
            cv_mses.append(mse)

        mean_cv_mse = np.mean(cv_mses)
        if mean_cv_mse < best_cv_mse:
            best_cv_mse = mean_cv_mse
            best_lambda = lam

    # Final fit using all data with the chosen lambda
    if time.time() < deadline - 5.0:
        A = V.T @ V + best_lambda * np.eye(6)
        b = V.T @ y
        try:
            c_scaled = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            c_scaled = np.linalg.lstsq(V, y, rcond=None)[0]
    else:
        # Fallback: unregularized least squares
        c_scaled = np.linalg.lstsq(V, y, rcond=None)[0]

    # Transform coefficients from scaled basis back to original x basis
    # We have: p(x_scaled) = sum_{i=0}^{5} c_scaled[i] * x_scaled^i
    # where x_scaled = (x - x_mean) / x_std
    # We need p(x) = sum_{j=0}^{5} c_orig[j] * x^j
    #
    # x_scaled^i = ((x - mu)/s)^i = sum_{j=0}^{i} binom(i,j) * x^j * (-mu/s)^(i-j)
    # So c_orig[j] = sum_{i=j}^{5} c_scaled[i] * binom(i,j) * (-mu/s)^(i-j)

    mu = x_mean
    s = x_std
    c_orig = np.zeros(6)
    for i in range(6):
        for j in range(i + 1):
            from math import comb
            coeff_term = c_scaled[i] * comb(i, j) * ((-mu / s) ** (i - j))
            c_orig[j] += coeff_term

    # Verify finiteness
    if not np.all(np.isfinite(c_orig)):
        # Fallback to simple linear fit
        c_orig = np.zeros(6)
        try:
            c_orig[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except Exception:
            pass

    return c_orig
# EVOLVE-BLOCK-END
