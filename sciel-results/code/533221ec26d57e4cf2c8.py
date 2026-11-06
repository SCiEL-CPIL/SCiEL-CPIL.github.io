"""Climate dynamics: unresolved-tendency closure via Tikhon-regularized polynomial fitting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _solve_ridge(VtV, Vty, lam, n_coeffs):
    """Solve ridge regression: (V^T V + lam * I) c = V^T y."""
    A = VtV + lam * np.eye(n_coeffs)
    try:
        return np.linalg.solve(A, Vty)
    except np.linalg.LinAlgError:
        try:
            return np.linalg.lstsq(A, Vty, rcond=None)[0]
        except Exception:
            return np.zeros(n_coeffs)


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved subgrid tendency.
    Uses Tikhonov (L2) regularization with lambda selected by internal k-fold
    cross-validation to control overfitting and improve generalization to
    unseen forcing regimes (e.g., F=10).
    
    The polynomial is fit in a scaled basis for numerical stability, then
    converted back to coefficients in ascending powers of the original x.
    """
    t0 = time.time()
    DEADLINE = 120.0  # seconds; leave margin for evaluator overhead
    SAFETY_MARGIN = 8.0  # return at least 8s before deadline

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    # Remove any NaN/Inf
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    n = len(x)

    if n < 6:
        return np.zeros(6)

    # Check time early
    if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, min(5, n - 1))
            out = np.zeros(6)
            out[:len(coeff)] = coeff
            return out
        except Exception:
            return np.zeros(6)

    # Scale x to [-1, 1] for better conditioning
    x_min = x.min()
    x_max = x.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        c = np.zeros(6)
        c[0] = y.mean()
        return c

    # Map x to z in [-1, 1]
    z = 2.0 * (x - x_min) / x_range - 1.0

    # Build Vandermonde matrix in scaled basis: columns z^0, z^1, ..., z^5
    V = np.vander(z, N=6, increasing=True)  # n x 6

    # Cross-validation setup
    n_folds = 10
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)

    # Precompute fold data: for each fold, store VtV, Vty for training set
    fold_data = []
    for f in range(n_folds):
        val_idx = folds[f]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != f])

        V_train = V[train_idx]
        y_train = y[train_idx]
        V_val = V[val_idx]
        y_val = y[val_idx]

        VtV = V_train.T @ V_train  # 6x6
        Vty = V_train.T @ y_train  # 6x1

        fold_data.append((VtV, Vty, V_val, y_val))

    def compute_cv_score(lam):
        """Compute mean CV MSE for a given lambda."""
        cv_errors = []
        for VtV, Vty, V_val, y_val in fold_data:
            c_train = _solve_ridge(VtV, Vty, lam, 6)
            pred_val = V_val @ c_train
            cv_errors.append(np.mean((pred_val - y_val) ** 2))
        if cv_errors:
            return np.mean(cv_errors)
        return np.inf

    # Coarse lambda search
    lambdas_coarse = np.concatenate([
        [0.0],
        np.logspace(-12, 8, 50)
    ])

    best_lambda = 0.0
    best_cv_score = np.inf

    for lam in lambdas_coarse:
        if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
            break
        cv_score = compute_cv_score(lam)
        if cv_score < best_cv_score:
            best_cv_score = cv_score
            best_lambda = lam

    # Fine lambda search around best_lambda
    if best_lambda > 0:
        lower = max(best_lambda * 0.3, 1e-15)
        upper = best_lambda * 3.0
        lambdas_fine = np.logspace(np.log10(lower), np.log10(upper), 25)
    else:
        lambdas_fine = np.logspace(-12, -1, 25)

    for lam in lambdas_fine:
        if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
            break
        cv_score = compute_cv_score(lam)
        if cv_score < best_cv_score:
            best_cv_score = cv_score
            best_lambda = lam

    # Final fit on all data with best lambda
    VtV_full = V.T @ V
    Vty_full = V.T @ y

    if best_lambda < 1e-14:
        lam_final = 1e-14
    else:
        lam_final = best_lambda

    coeff_z = _solve_ridge(VtV_full, Vty_full, lam_final, 6)

    # Check time
    if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
        # Convert and return quickly
        pass

    # Convert from scaled basis z = 2*(x - x_min)/x_range - 1 to original x
    # z = (2/x_range) * x + (-2*x_min/x_range - 1)
    # Let a = 2/x_range, b = -2*x_min/x_range - 1
    # z = a*x + b
    # We have y = sum_{i=0}^{5} c_i * z^i = sum_{i=0}^{5} c_i * (a*x + b)^i
    # Expand (a*x + b)^i = sum_{j=0}^{i} C(i,j) * (a*x)^j * b^{i-j}
    #                   = sum_{j=0}^{i} C(i,j) * a^j * b^{i-j} * x^j
    # So y = sum_{j=0}^{5} x^j * [sum_{i=j}^{5} c_i * C(i,j) * a^j * b^{i-j}]

    a = 2.0 / x_range
    b = -2.0 * x_min / x_range - 1.0

    orig_coeff = np.zeros(6)
    for j in range(6):
        s = 0.0
        for i in range(j, 6):
            # C(i, j)
            binom = 1.0
            for k in range(1, j + 1):
                binom *= (i - k + 1) / k
            s += coeff_z[i] * binom * (a ** j) * (b ** (i - j))
        orig_coeff[j] = s

    # Final check for finite values
    if not np.all(np.isfinite(orig_coeff)):
        # Fall back to simple polyfit
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, min(5, n - 1))
            out = np.zeros(6)
            out[:len(coeff)] = coeff
            return out
        except Exception:
            return np.zeros(6)

    return orig_coeff
# EVOLVE-BLOCK-END
