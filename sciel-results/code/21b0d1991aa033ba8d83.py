"""Rewritten program: odd-power polynomial closure with CV-regularized Tikhonov fit."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin before 150s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Filter non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 10:
        # Not enough data; return zero coefficients
        return np.zeros(6)

    # The subgrid tendency h*c*mean(Y) is approximately an odd function of X
    # due to the symmetry of the Lorenz-96 fast dynamics.
    # Fit using only odd powers: x, x^3, x^5
    # This reduces parameters from 6 to 3, improving generalization.

    # Build design matrix with odd powers
    A = np.column_stack([x, x**3, x**5])
    n = x.size

    # Determine number of folds for cross-validation
    # Use 5-fold CV if enough samples, else leave-one-out is too slow
    if n >= 50:
        n_folds = 5
    elif n >= 20:
        n_folds = 4
    else:
        n_folds = 3

    # Candidate regularization strengths (log-spaced)
    # Scale lambda relative to data magnitude for meaningful regularization
    y_scale = np.std(y)
    if y_scale == 0 or not np.isfinite(y_scale):
        y_scale = 1.0

    # Lambda range: from very small to moderately large relative to data scale
    # A = A.T @ A has magnitude ~ n * mean(x^2), so lambda should be comparable
    gram_scale = np.mean(np.sum(A**2, axis=0))
    if gram_scale == 0 or not np.isfinite(gram_scale):
        gram_scale = 1.0

    # Log-spaced lambdas from 1e-8 to 1e2 times gram_scale
    log_lam_min = np.log10(gram_scale) - 8
    log_lam_max = np.log10(gram_scale) + 2
    n_lam = 25
    log_lams = np.linspace(log_lam_min, log_lam_max, n_lam)
    lams = 10**log_lams

    # K-fold cross-validation to select best lambda
    # Shuffle indices once for consistent folds
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)

    # Precompute fold masks
    fold_sizes = np.array([n // n_folds] * n_folds)
    fold_sizes[:n % n_folds] += 1
    fold_starts = np.concatenate([[0], np.cumsum(fold_sizes)])

    best_cv_mse = np.inf
    best_lam = lams[0]
    best_coeff_odd = None

    for i_lam, lam in enumerate(lams):
        # Check time budget
        if time.time() > deadline - 5.0:
            break

        cv_mse_sum = 0.0
        cv_count = 0

        for k in range(n_folds):
            # Test fold
            test_idx = indices[fold_starts[k]:fold_starts[k+1]]
            # Train fold: all other indices
            train_idx = np.concatenate([indices[:fold_starts[k]], indices[fold_starts[k+1]:]])

            A_train = A[train_idx]
            y_train = y[train_idx]
            A_test = A[test_idx]
            y_test = y[test_idx]

            # Solve regularized least squares: (A^T A + lam * I) c = A^T y
            try:
                gram = A_train.T @ A_train
                rhs = A_train.T @ y_train
                coeff_k = np.linalg.solve(gram + lam * np.eye(3), rhs)
            except np.linalg.LinAlgError:
                try:
                    coeff_k = np.linalg.lstsq(A_train, y_train, rcond=None)[0]
                except:
                    continue

            # Compute MSE on test fold
            pred = A_test @ coeff_k
            mse = np.mean((pred - y_test)**2)
            cv_mse_sum += mse
            cv_count += 1

        if cv_count > 0:
            cv_mse = cv_mse_sum / cv_count
            if cv_mse < best_cv_mse:
                best_cv_mse = cv_mse
                best_lam = lam
                # Store the coefficient vector for this lambda (will refit on full data)
                best_coeff_odd = coeff_k  # placeholder, we refit below

    # Check time budget
    if time.time() > deadline - 3.0:
        # Use the best lambda found so far, refit on full data
        try:
            gram = A.T @ A
            rhs = A.T @ y
            coeff_odd = np.linalg.solve(gram + best_lam * np.eye(3), rhs)
        except np.linalg.LinAlgError:
            coeff_odd = np.linalg.lstsq(A, y, rcond=None)[0]
    else:
        # Refit on full dataset with best lambda
        try:
            gram = A.T @ A
            rhs = A.T @ y
            coeff_odd = np.linalg.solve(gram + best_lam * np.eye(3), rhs)
        except np.linalg.LinAlgError:
            try:
                coeff_odd = np.linalg.lstsq(A, y, rcond=None)[0]
            except:
                # Ultimate fallback: unregularized least squares
                coeff_odd = np.zeros(3)
                coeff_odd[0] = np.dot(x, y) / (np.dot(x, x) + 1e-15)

    # Map odd coefficients to full degree-5 polynomial
    coeff = np.zeros(6)
    coeff[1] = coeff_odd[0]  # x^1
    coeff[3] = coeff_odd[1]  # x^3
    coeff[5] = coeff_odd[2]  # x^5

    # Verify finite
    if not np.all(np.isfinite(coeff)):
        # Fallback to simple linear
        try:
            c1 = np.dot(x, y) / (np.dot(x, x) + 1e-15)
            coeff = np.zeros(6)
            coeff[1] = c1
        except:
            coeff = np.zeros(6)

    return coeff
# EVOLVE-BLOCK-END
