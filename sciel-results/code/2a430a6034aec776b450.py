import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 110.0  # Safety margin

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)

    x = x[:n]
    y = y[:n]

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size

    if n == 0:
        return np.zeros(6, dtype=np.float64)

    degree = 5

    # Use a wider range of lambda values for better regularization search
    lambdas = np.logspace(-10, 8, 80)

    # Cross-Validation to find best lambda
    k_folds = 5
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_sizes = np.full(k_folds, n // k_folds, dtype=int)
    fold_sizes[:n % k_folds] += 1
    folds = np.split(indices, np.cumsum(fold_sizes)[:-1])

    best_lambda = 1.0
    best_cv_mse = np.inf

    for lam in lambdas:
        if time.time() - t_start > deadline - 10:
            break

        mse_sum = 0.0
        for i in range(k_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])

            x_train = x[train_idx]
            y_train = y[train_idx]
            x_val = x[val_idx]
            y_val = y[val_idx]

            # Fit Ridge directly in original x space
            V_train = np.vander(x_train, N=degree + 1, increasing=True)
            A = V_train.T @ V_train + lam * np.eye(degree + 1)
            b = V_train.T @ y_train

            try:
                coeff = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                coeff = np.linalg.lstsq(A, b, rcond=None)[0]

            V_val = np.vander(x_val, N=degree + 1, increasing=True)
            pred = V_val @ coeff
            mse = np.mean((pred - y_val) ** 2)
            mse_sum += mse

        avg_mse = mse_sum / k_folds
        if avg_mse < best_cv_mse:
            best_cv_mse = avg_mse
            best_lambda = lam

    # Final fit on all data with best_lambda
    V_all = np.vander(x, N=degree + 1, increasing=True)
    A = V_all.T @ V_all + best_lambda * np.eye(degree + 1)
    b = V_all.T @ y

    try:
        coeff_orig = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        coeff_orig = np.linalg.lstsq(A, b, rcond=None)[0]

    if not np.all(np.isfinite(coeff_orig)):
        coeff_orig = np.nan_to_num(coeff_orig, nan=0.0, posinf=0.0, neginf=0.0)

    return coeff_orig
